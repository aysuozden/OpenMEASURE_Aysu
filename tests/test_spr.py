import pytest
import src.openmeasure.sparse_sensing as sps
import numpy as np

class TestSPR:

    def setup_method(self, method):
        rng = np.random.default_rng()
        self.n_points = 10
        self.n_features = 2
        self.m = 5
        X = rng.random(size=(self.n_points*self.n_features, self.m))   
        xyz = rng.random(size=(self.n_points, 3))
        self.C = np.eye(X.shape[0])
        self.spr = sps.SPR(X, self.n_features, xyz)
        

    def teardown_method(self, method):
        pass

    def test_optimal_placement_qr(self):
        self.spr.fit(n_modes=100)
        C_qr = self.spr.optimal_placement()
        assert C_qr.shape[0] == self.m
        assert C_qr.shape[1] == self.spr.X.shape[0]

    def test_scale_vector(self):
        X_cnt = np.mean(self.spr.X, axis=1)[:, np.newaxis]
        
        self.spr.fit(n_modes=100)
        self.spr.train(self.C)
        
        y = np.zeros((self.C.shape[0], 2))
        y[:,0] = self.C @ self.spr.X[:, 0]
        
        y0 = self.spr.scale_vector(y)

        y0_check = np.zeros((self.C.shape[0], 2))
        y0_check[:,0] = (y[:,0]-X_cnt[:,0])
        
        np.testing.assert_allclose(y0, y0_check)
    
    def test_predict(self):
        self.spr.fit(n_modes=100)
        self.spr.train(self.C)

        y = np.zeros((self.C.shape[0], 2))
        y[:,0] = self.C @ self.spr.X[:, 0]
        
        a, _ = self.spr.predict(y)
        x_pred = self.spr.reconstruct(a)

        np.testing.assert_allclose(x_pred, self.spr.X[:, [0]])        

    def test_predict_uncertainty(self):
        self.spr.fit(n_modes=100)
        self.spr.train(self.C)

        y = np.zeros((self.C.shape[0], 2))
        y[:,0] = self.C @ self.spr.X[:, 0]
        y[:,1] = 0.1  
        
        _, a_cov = self.spr.predict(y)
        
        L = np.linalg.cholesky(a_cov[0,:,:]) 
        