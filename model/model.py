import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, input_channel, patch_size, embedding_dim, 
                       num_patch):
        super().__init__()
        self.patch_layer = nn.Conv2d(input_channel, embedding_dim, 
                                     kernel_size = patch_size, 
                                     stride = patch_size)
        
        self.embedding_dim = embedding_dim
        
        # The class token is inherently the embedding of the image that goes into the final MLP layer
        self.class_token = nn.Parameter(torch.randn(embedding_dim, 1))
        
        # Position embedding is applied to capture the relative position of the patches in the image
        self.positional_embedding = nn.Parameter(torch.randn(embedding_dim, num_patch + 1))

    def forward(self, x):
        x = self.patch_layer(x) # batch_size x embedding_dim x height x width
        x = x.flatten(start_dim = -2) # batch_size x embedding_dim x (height x width)
        class_token = self.class_token.expand(x.size(0), self.embedding_dim, 1)
        x = torch.cat([class_token, x], dim = -1) # batch_size x embedding_dim x (height x width + 1)
        x  = x + self.positional_embedding
        x = torch.transpose(x, -2, -1) #batch_size x (height x width + 1)x embedding_dim
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, projection_dim, num_heads = 1):
        """
        This forms the core of the Transformer encoder. Using the nn.MultiheadAttention really simplifies
        the actual implementation of the attention. 

        The transformer encoder stack comprises of the attention part followed by the MLP or a linear layers
        with a GeLU activation. 

        The transformer stack is usually increased on top of each other
        """
        super().__init__()
        # Normalizing layer that normalizes along the last dimension
        self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)
        
        # Pytorch MultiheadAttention really simplifies the actual coding of attention part
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first = True)

        # A linear transformation is also applied 
        self.mlp_block = nn.Sequential(*[
            nn.Linear(embedding_dim, projection_dim), 
            nn.GELU(), 
            nn.Linear(projection_dim, embedding_dim),
        ])

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.self_attention(x,x,x)[0] + x
        x = self.mlp_block(x) + x
        
        return x

class AttentionModel(nn.Module):
    def __init__(self, patch_size, 
                 output_dim, NUM_PATCH,
                 L = 2, input_channel = 1, 
                 embedding_dim = 16, num_heads = 2):
        """
        Forms the entire Visual Transformer model. 
        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(input_channel, 
                                              patch_size, embedding_dim, 
                                              NUM_PATCH)

        # Mutliple layers of the transformer encoder is stacked on top of another.
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(embedding_dim, 
                                                                      projection_dim = embedding_dim,
                                                                      num_heads = num_heads) for _ in range(L)])

        # The last layer takes the class token and gives the probability over the classes
        self.output_layer = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)            
        x = x[:, 0, :] # Extracting the class token
        x = self.output_layer(x)
        return x