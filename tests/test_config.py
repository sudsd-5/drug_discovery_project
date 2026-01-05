import pytest
import yaml
import os
from pathlib import Path


class TestConfiguration:
    
    def test_config_file_exists(self):
        """Test that the main config file exists"""
        config_path = Path("dti_config.yaml")
        assert config_path.exists(), "Config file dti_config.yaml not found"
    
    def test_config_loading(self):
        """Test loading configuration from YAML"""
        config_path = Path("dti_config.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert isinstance(config, dict)
    
    def test_config_structure(self):
        """Test configuration structure"""
        config_path = Path("dti_config.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['data', 'model', 'training', 'hardware', 'logging']
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"
    
    def test_model_parameters(self):
        """Test model parameter values"""
        config_path = Path("dti_config.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config.get('model', {})
        
        assert 'drug_input_dim' in model_config
        assert 'protein_input_dim' in model_config
        assert model_config['drug_input_dim'] > 0
        assert model_config['protein_input_dim'] > 0
    
    def test_training_parameters(self):
        """Test training parameter values"""
        config_path = Path("dti_config.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        training_config = config.get('training', {})
        
        assert 'batch_size' in training_config
        assert 'epochs' in training_config
        assert 'learning_rate' in training_config
        
        assert training_config['batch_size'] > 0
        assert training_config['epochs'] > 0
        assert 0 < training_config['learning_rate'] < 1
    
    def test_relative_paths(self):
        """Test that configuration uses relative paths"""
        config_path = Path("dti_config.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_config = config.get('data', {})
        
        if 'interactions_dir' in data_config:
            path = data_config['interactions_dir']
            # Should not start with drive letter (Windows) or be absolute
            assert not path.startswith('C:'), "Path should be relative, not absolute"
            assert not path.startswith('/home'), "Path should be relative to project root"
    
    def test_device_configuration(self):
        """Test device configuration"""
        config_path = Path("dti_config.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        hardware_config = config.get('hardware', {})
        
        assert 'device' in hardware_config
        assert hardware_config['device'] in ['cuda', 'cpu']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
