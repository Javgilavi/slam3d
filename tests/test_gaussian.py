import numpy as np
import torch

from slam3d.dense.gaussian import composite, projected_covariance, raster_table


def test_covariance_jacobian_against_finite_difference():
    p=np.array([[.3,-.2,2.]])
    C=np.array([[[.1,.02,0],[.02,.03,.01],[0,.01,.05]]])
    def project(v): return 100*v[:2]/v[2]
    h=1e-5
    J=np.column_stack([(project(p[0]+np.eye(3)[i]*h)-project(p[0]-np.eye(3)[i]*h))/(2*h) for i in range(3)])
    np.testing.assert_allclose(projected_covariance(p,C,100)[0],J@C[0]@J.T+np.eye(2)*.3,rtol=1e-7)


def test_front_to_back_alpha_and_gradients():
    rgb=torch.tensor([[1.,0,0],[0,0,1.]],requires_grad=True)
    opacity=torch.tensor([.5,.5],requires_grad=True)
    color,alpha=composite(rgb,opacity,torch.tensor([[0,1]]),torch.ones(1,2))
    np.testing.assert_allclose(color.detach(),[[.5,0,.25]])
    np.testing.assert_allclose(alpha.detach(),[.75])
    color.sum().backward()
    assert torch.isfinite(opacity.grad).all() and rgb.grad.abs().sum()>0


def test_raster_depth_order_and_behind_camera_rejection():
    xyz=np.array([[0.,0,3],[0,0,1],[0,0,-1]])
    cov=np.repeat(np.eye(3)[None]*.01,3,axis=0)
    ids,w,_=raster_table(xyz,cov,np.eye(4),size=32,layers=8)
    row=16*32+16
    assert ids[row,0]==1 and ids[row,1]==0
    assert not (ids[w>0]==2).any()


def test_raster_invariant_under_shared_world_transform():
    from scipy.spatial.transform import Rotation
    xyz=np.array([[.1,.1,2.],[-.2,.1,1.3]])
    C=np.repeat(np.diag([.01,.02,.005])[None],2,axis=0)
    T=np.eye(4);ids,w,_=raster_table(xyz,C,T,size=32)
    G=np.eye(4);G[:3,:3]=Rotation.from_rotvec([.4,-.2,.1]).as_matrix();G[:3,3]=[1,2,3]
    ids2,w2,_=raster_table(xyz@G[:3,:3].T+G[:3,3],G[:3,:3]@C@G[:3,:3].T,G,size=32)
    np.testing.assert_array_equal(ids,ids2)
    np.testing.assert_allclose(w,w2,atol=1e-6)
