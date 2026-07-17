import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBiLSTMEmotionModel(nn.Module):
    """
    Hybrid CNN-BiLSTM model fusing GloVe text embeddings with Emoji2Vec emoji embeddings.

    Text branch: CNN (local n-gram features) -> BiLSTM (sequential context) -> pooled vector
    Emoji branch: mean-pooled Emoji2Vec vectors -> dense projection
    Fusion: concatenate both -> fully connected classifier
    """

    def __init__(
        self,
        text_embed_dim=100,
        emoji_embed_dim=300,
        cnn_out_channels=128,
        cnn_kernel_sizes=(3, 4, 5),
        lstm_hidden_dim=128,
        emoji_proj_dim=64,
        num_classes=28,       # GoEmotions has 27 emotions + neutral
        dropout=0.3
    ):
        super().__init__()

# --- Text branch: CNN ---
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=text_embed_dim, out_channels=cnn_out_channels, kernel_size=k, padding='same')
            for k in cnn_kernel_sizes
        ])
        # --- Text branch: BiLSTM (operates on concatenated CNN feature maps) ---
        cnn_total_out = cnn_out_channels * len(cnn_kernel_sizes)
        self.bilstm = nn.LSTM(
            input_size=cnn_total_out,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # --- Emoji branch: simple projection after mean pooling ---
        self.emoji_proj = nn.Sequential(
            nn.Linear(emoji_embed_dim, emoji_proj_dim),
            nn.ReLU()
        )

        # --- Fusion + classifier ---
        fusion_dim = (lstm_hidden_dim * 2) + emoji_proj_dim  # *2 for bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, text_embeds, emoji_embeds):
        """
        text_embeds: (batch, seq_len, text_embed_dim)
        emoji_embeds: (batch, num_emojis, emoji_embed_dim)
        """
        # --- Text branch ---
        x = text_embeds.permute(0, 2, 1)  # -> (batch, embed_dim, seq_len) for Conv1d
        conv_outs = [F.relu(conv(x)) for conv in self.convs]           # each: (batch, cnn_out_channels, seq_len)
        conv_outs = [c.permute(0, 2, 1) for c in conv_outs]            # -> (batch, seq_len, cnn_out_channels)
        cnn_features = torch.cat(conv_outs, dim=2)                     # -> (batch, seq_len, cnn_total_out)

        lstm_out, _ = self.bilstm(cnn_features)                        # -> (batch, seq_len, hidden*2)
        text_vector = torch.mean(lstm_out, dim=1)                      # mean pooling -> (batch, hidden*2)

        # --- Emoji branch ---
        emoji_pooled = torch.mean(emoji_embeds, dim=1)                 # (batch, emoji_embed_dim)
        emoji_vector = self.emoji_proj(emoji_pooled)                   # (batch, emoji_proj_dim)

        # --- Fusion ---
        fused = torch.cat([text_vector, emoji_vector], dim=1)          # (batch, fusion_dim)
        logits = self.classifier(fused)                                # (batch, num_classes)

        return logits


if __name__ == "__main__":
    # Quick sanity test with dummy data
    batch_size = 4
    seq_len = 20
    num_emojis = 3

    dummy_text = torch.randn(batch_size, seq_len, 100)
    dummy_emoji = torch.randn(batch_size, num_emojis, 300)

    model = CNNBiLSTMEmotionModel()
    output = model(dummy_text, dummy_emoji)

    print("Output shape:", output.shape)  # expected: (4, 28)