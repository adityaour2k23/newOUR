import torch
from torch_geometric.data import Data
        
class GNNTransformMD(object):
    """
    Transform the dict returned by the ProtDataset class to a pyTorch Geometric graph
    """

    def __init__(self, edge_dist_cutoff=4.5):
        """

        Args:
            edge_dist_cutoff (float, optional): distence between the edges. Defaults to 4.5.
        """
        self.edge_dist_cutoff = edge_dist_cutoff 

    def __call__(self, data_dict):

        score_features = data_dict["scores"]
    
        #protein_features = data_dict["atoms_protein"]
      
        frames_features = data_dict["frames"]
        
        score_features_tensor = torch.tensor(score_features, dtype=torch.float)

        #protein_features_tensor = torch.tensor(protein_features, dtype=torch.float)

        frames_features_tensor = torch.tensor(frames_features, dtype=torch.float)

         # Create a PyTorch Geometric Data object
        data = Data(
            score=score_features_tensor,
            #protein=protein_features_tensor,  
            frames=frames_features_tensor,  
            pid=data_dict["id"]
        )

        return data
