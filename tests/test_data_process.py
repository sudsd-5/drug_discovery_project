import pytest
import torch
import numpy as np
from rdkit import Chem


class TestDataProcessing:
    
    def test_smiles_to_mol(self):
        """Test SMILES string conversion to RDKit molecule"""
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert mol.GetNumAtoms() == 3
    
    def test_invalid_smiles(self):
        """Test handling of invalid SMILES"""
        invalid_smiles = "INVALID_SMILES_123"
        mol = Chem.MolFromSmiles(invalid_smiles)
        assert mol is None
    
    def test_mol_features(self):
        """Test molecular feature extraction"""
        smiles = "CC(=O)O"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        assert num_atoms > 0
        assert num_bonds > 0
    
    def test_protein_sequence_validation(self):
        """Test protein sequence validation"""
        valid_seq = "MKFLILLFNILCLFPVLA"
        amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        
        assert all(aa in amino_acids for aa in valid_seq)
        assert len(valid_seq) > 0


class TestTensorOperations:
    
    def test_tensor_creation(self):
        """Test PyTorch tensor creation"""
        data = [[1, 2, 3], [4, 5, 6]]
        tensor = torch.tensor(data, dtype=torch.float32)
        
        assert tensor.shape == (2, 3)
        assert tensor.dtype == torch.float32
    
    def test_tensor_operations(self):
        """Test basic tensor operations"""
        x = torch.randn(10, 5)
        y = torch.randn(10, 5)
        
        z = x + y
        assert z.shape == (10, 5)
        
        w = torch.matmul(x, y.t())
        assert w.shape == (10, 10)
    
    def test_cuda_availability(self):
        """Test CUDA availability (will pass on CPU-only systems)"""
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            device = torch.device("cuda")
            x = torch.randn(5, 5, device=device)
            assert x.is_cuda
        else:
            assert True


class TestModelComponents:
    
    def test_batch_processing(self):
        """Test batch data processing"""
        batch_size = 32
        feature_dim = 128
        
        batch = torch.randn(batch_size, feature_dim)
        assert batch.shape == (batch_size, feature_dim)
    
    def test_activation_functions(self):
        """Test activation functions"""
        x = torch.randn(10, 10)
        
        relu_out = torch.relu(x)
        assert relu_out.shape == x.shape
        assert (relu_out >= 0).all()
        
        sigmoid_out = torch.sigmoid(x)
        assert sigmoid_out.shape == x.shape
        assert ((sigmoid_out >= 0) & (sigmoid_out <= 1)).all()
    
    def test_loss_computation(self):
        """Test loss computation"""
        pred = torch.randn(32, 1)
        target = torch.randint(0, 2, (32, 1)).float()
        
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
