// slam3d driver for stella_vslam (upstream library used unmodified).
//
// Differences from stella_vslam_examples/run_image_slam:
//   * feeds the TRUE per-frame timestamps from frames.csv (the example synthesises 1/fps steps);
//   * logs the ONLINE pose returned for every frame (before later loop-closure corrections),
//     the tracking state and whether loop BA is running, so downstream localization can be
//     evaluated causally and global corrections are not mistaken for motion;
//   * logs, for every k-th tracked frame, the landmarks tracked in that frame expressed in the
//     CURRENT camera frame using the online pose (a causal local-structure observation);
//   * exports final keyframes, landmarks and keyframe->landmark observations in simple binary/CSV.
//
// Output files (all little-endian):
//   online_poses.txt   idx timestamp state loop_ba n_tracked track_ms tx ty tz qx qy qz qw  (T_world_cam; NaN if lost)
//   local_obs.bin      repeated: int32 idx, float64 t, int32 n, n * (uint32 lm_id, float32 xc, yc, zc)
//   keyframes.csv      id,timestamp,r00..r22,tx,ty,tz  (T_world_cam)
//   landmarks.bin      repeated: uint32 id, float32 x y z, uint32 num_obs, uint32 first_kf
//   kf_obs.bin         repeated: uint32 kf_id, uint32 lm_id, float32 u, float32 v   (undistorted keypoint px)
//   frame_trajectory.txt / keyframe_trajectory.txt (TUM, globally optimised), map.msg
#include <stella_vslam/system.h>
#include <stella_vslam/config.h>
#include <stella_vslam/data/keyframe.h>
#include <stella_vslam/data/landmark.h>
#include <stella_vslam/publish/frame_publisher.h>
#include <stella_vslam/publish/map_publisher.h>

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Args {
    std::string vocab, config, frames, mask, out;
    int obs_every = 1;
    int max_frames = -1;
    int start = 0;
    double throttle_ms = 0.0;
};

void usage() {
    std::cerr << "slam3d_stella_driver --vocab V --config C --frames frames.csv --out DIR "
                 "[--mask M] [--obs-every K] [--start I] [--max-frames N] [--throttle-ms MS]\n";
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--vocab") a.vocab = next();
        else if (k == "--config") a.config = next();
        else if (k == "--frames") a.frames = next();
        else if (k == "--mask") a.mask = next();
        else if (k == "--out") a.out = next();
        else if (k == "--obs-every") a.obs_every = std::stoi(next());
        else if (k == "--max-frames") a.max_frames = std::stoi(next());
        else if (k == "--start") a.start = std::stoi(next());
        else if (k == "--throttle-ms") a.throttle_ms = std::stod(next());
        else { std::cerr << "unknown arg " << k << "\n"; return false; }
    }
    return !a.vocab.empty() && !a.config.empty() && !a.frames.empty() && !a.out.empty();
}

struct FrameRow {
    int idx;
    double t;
    std::string path;
};

std::vector<FrameRow> read_frames(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<FrameRow> rows;
    std::string line;
    std::string dir = path.substr(0, path.find_last_of('/') + 1);
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("idx", 0) == 0) continue;
        std::stringstream ss(line);
        std::string a, b, c;
        std::getline(ss, a, ',');
        std::getline(ss, b, ',');
        std::getline(ss, c, ',');
        FrameRow r{std::stoi(a), std::stod(b), c};
        if (!r.path.empty() && r.path[0] != '/') r.path = dir + r.path;
        rows.push_back(r);
    }
    return rows;
}

void write_pose_line(std::ofstream& o, int idx, double t, const std::string& state, bool loop_ba,
                     size_t n_tracked, double ms, const std::shared_ptr<stella_vslam::Mat44_t>& pose_wc) {
    o << idx << " " << std::setprecision(15) << t << " " << state << " " << (loop_ba ? 1 : 0) << " "
      << n_tracked << " " << std::setprecision(6) << ms << " ";
    if (pose_wc) {
        Eigen::Matrix3d R = pose_wc->block<3, 3>(0, 0);
        Eigen::Vector3d tr = pose_wc->block<3, 1>(0, 3);
        Eigen::Quaterniond q(R);
        o << std::setprecision(9) << tr.x() << " " << tr.y() << " " << tr.z() << " " << q.x() << " " << q.y()
          << " " << q.z() << " " << q.w() << "\n";
    }
    else {
        o << "nan nan nan nan nan nan nan\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    try {
        if (!parse(argc, argv, a)) { usage(); return 2; }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        usage();
        return 2;
    }
    const auto rows = read_frames(a.frames);
    std::cerr << "[driver] " << rows.size() << " frames listed\n";

    auto cfg = std::make_shared<stella_vslam::config>(a.config);
    auto slam = std::make_shared<stella_vslam::system>(cfg, a.vocab);
    slam->startup();

    cv::Mat mask;
    if (!a.mask.empty()) mask = cv::imread(a.mask, cv::IMREAD_GRAYSCALE);

    std::ofstream poses(a.out + "/online_poses.txt");
    poses << "# idx timestamp state loop_ba n_tracked track_ms tx ty tz qx qy qz qw (T_world_cam, online)\n";
    std::ofstream obs(a.out + "/local_obs.bin", std::ios::binary);

    auto fpub = slam->get_frame_publisher();
    int fed = 0;
    const auto t_start = std::chrono::steady_clock::now();
    for (size_t i = static_cast<size_t>(std::max(0, a.start)); i < rows.size(); ++i) {
        if (a.max_frames > 0 && fed >= a.max_frames) break;
        const auto& r = rows[i];
        cv::Mat img = cv::imread(r.path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "[driver] cannot read " << r.path << "\n";
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        auto pose_wc = slam->feed_monocular_frame(img, r.t, mask);
        const auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const std::string state = fpub->get_tracking_state();
        const bool loop_ba = slam->loop_BA_is_running();

        size_t n_tracked = 0;
        std::vector<std::pair<unsigned int, Eigen::Vector3f>> local;
        if (pose_wc) {
            const auto lms = fpub->get_landmarks();
            const Eigen::Matrix4d cw = pose_wc->inverse();
            const Eigen::Matrix3d Rcw = cw.block<3, 3>(0, 0);
            const Eigen::Vector3d tcw = cw.block<3, 1>(0, 3);
            for (const auto& lm : lms) {
                if (!lm || lm->will_be_erased()) continue;
                ++n_tracked;
                if (a.obs_every > 0 && (r.idx % a.obs_every) == 0) {
                    Eigen::Vector3d pc = Rcw * lm->get_pos_in_world() + tcw;
                    local.emplace_back(lm->id_, pc.cast<float>());
                }
            }
        }
        write_pose_line(poses, r.idx, r.t, state, loop_ba, n_tracked, ms, pose_wc);
        if (!local.empty()) {
            int32_t idx = r.idx;
            double t = r.t;
            int32_t n = static_cast<int32_t>(local.size());
            obs.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
            obs.write(reinterpret_cast<const char*>(&t), sizeof(t));
            obs.write(reinterpret_cast<const char*>(&n), sizeof(n));
            for (const auto& [id, p] : local) {
                uint32_t uid = id;
                obs.write(reinterpret_cast<const char*>(&uid), sizeof(uid));
                obs.write(reinterpret_cast<const char*>(p.data()), 3 * sizeof(float));
            }
        }
        ++fed;
        if (fed % 200 == 0) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            std::cerr << "[driver] frame " << r.idx << " state=" << state << " tracked=" << n_tracked
                      << " fps=" << fed / el << "\n";
        }
        if (a.throttle_ms > 0) std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(a.throttle_ms));
    }
    poses.close();
    obs.close();

    std::cerr << "[driver] waiting for loop BA\n";
    while (slam->loop_BA_is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(50));

    slam->save_frame_trajectory(a.out + "/frame_trajectory.txt", "TUM");
    slam->save_keyframe_trajectory(a.out + "/keyframe_trajectory.txt", "TUM");

    auto mpub = slam->get_map_publisher();
    std::vector<std::shared_ptr<stella_vslam::data::keyframe>> kfs;
    mpub->get_keyframes(kfs);
    std::ofstream kcsv(a.out + "/keyframes.csv");
    kcsv << "id,timestamp,r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n";
    std::ofstream kobs(a.out + "/kf_obs.bin", std::ios::binary);
    for (const auto& kf : kfs) {
        if (!kf || kf->will_be_erased()) continue;
        const Eigen::Matrix4d wc = kf->get_pose_wc();
        kcsv << kf->id_ << "," << std::setprecision(15) << kf->timestamp_ << std::setprecision(9);
        for (int rr = 0; rr < 3; ++rr)
            for (int cc = 0; cc < 3; ++cc) kcsv << "," << wc(rr, cc);
        kcsv << "," << wc(0, 3) << "," << wc(1, 3) << "," << wc(2, 3) << "\n";
        const auto lms = kf->get_landmarks();
        const auto& kps = kf->frm_obs_.undist_keypts_;
        for (size_t k = 0; k < lms.size() && k < kps.size(); ++k) {
            const auto& lm = lms[k];
            if (!lm || lm->will_be_erased()) continue;
            uint32_t kid = kf->id_, lid = lm->id_;
            float uv[2] = {kps[k].pt.x, kps[k].pt.y};
            kobs.write(reinterpret_cast<const char*>(&kid), 4);
            kobs.write(reinterpret_cast<const char*>(&lid), 4);
            kobs.write(reinterpret_cast<const char*>(uv), 8);
        }
    }
    std::vector<std::shared_ptr<stella_vslam::data::landmark>> lms;
    std::set<std::shared_ptr<stella_vslam::data::landmark>> local_lms;
    mpub->get_landmarks(lms, local_lms);
    std::ofstream lbin(a.out + "/landmarks.bin", std::ios::binary);
    for (const auto& lm : lms) {
        if (!lm || lm->will_be_erased()) continue;
        uint32_t id = lm->id_;
        Eigen::Vector3f p = lm->get_pos_in_world().cast<float>();
        uint32_t nobs = lm->num_observations();
        uint32_t fk = lm->first_keyfrm_id_;
        lbin.write(reinterpret_cast<const char*>(&id), 4);
        lbin.write(reinterpret_cast<const char*>(p.data()), 12);
        lbin.write(reinterpret_cast<const char*>(&nobs), 4);
        lbin.write(reinterpret_cast<const char*>(&fk), 4);
    }
    std::cerr << "[driver] keyframes=" << kfs.size() << " landmarks=" << lms.size() << "\n";
    slam->save_map_database(a.out + "/map.msg");
    slam->shutdown();
    std::cerr << "[driver] done\n";
    return 0;
}
