import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        base_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.layer0 = nn.Sequential(*list(base_resnet.children())[:4])  # conv1 + bn + relu + maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        # Self-attention modules
        self.self_attention_layer1 = SelfAttention(256)
        self.self_attention_layer2 = SelfAttention(512)
        self.self_attention_layer3 = SelfAttention(1024)
        self.self_attention_layer4 = SelfAttention(2048)

        # Inter-item attention modules
        self.cross_attention = CrossAttention(2048)
        self.global_attention = GlobalAttention(2048)

        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.Tanh()
        )

        self.classifier = nn.Linear(128 * 7, 13)

    def forward_once(self, x):
        trace = {}

        fmap = self.layer0(x)
        trace['after_layer0'] = fmap.detach().cpu()

        fmap = self.layer1(fmap)
        fmap = self.self_attention_layer1(fmap)
        trace['after_attn1'] = fmap.detach().cpu()

        fmap = self.layer2(fmap)
        fmap = self.self_attention_layer2(fmap)
        trace['after_attn2'] = fmap.detach().cpu()

        fmap = self.layer3(fmap)
        fmap = self.self_attention_layer3(fmap)
        trace['after_attn3'] = fmap.detach().cpu()

        fmap = self.layer4(fmap)
        fmap = self.self_attention_layer4(fmap)
        trace['after_attn4'] = fmap.detach().cpu()

        pooled = torch.mean(fmap.view(fmap.size(0), fmap.size(1), -1), dim=2)
        trace['pooled_embedding'] = pooled.detach().cpu()

        return pooled, fmap, trace

    def forward(self, *inputs):
        embeddings = []
        fmap_list = []
        trace_outputs = []
        masks = []

        for img in inputs:
            blank_white = torch.allclose(img, torch.ones_like(img), atol=1e-5)
            blank_black = torch.allclose(img, torch.zeros_like(img), atol=1e-5)
            masks.append(0 if blank_white or blank_black else 1)

            embed, fmap, trace = self.forward_once(img)
            embeddings.append(embed)
            fmap_list.append(fmap)
            trace_outputs.append(trace)

        masks = torch.tensor(masks, dtype=torch.float32, device=embeddings[0].device)
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = embeddings * masks.unsqueeze(-1)

        cross_attn_out = self.cross_attention(embeddings, embeddings)
        trace_outputs.append({'after_cross_attn': cross_attn_out.detach().cpu()})

        global_attn_out = self.global_attention(cross_attn_out)
        trace_outputs.append({'after_global_attn': global_attn_out.detach().cpu()})

        refined_embeddings = global_attn_out.view(-1, 7, 2048)
        refined_embeddings = self.fc(refined_embeddings.view(-1, 2048))
        trace_outputs.append({'after_fc': refined_embeddings.detach().cpu()})

        refined_embeddings = refined_embeddings.view(-1, 7 * 128)
        logits = self.classifier(refined_embeddings)
        trace_outputs.append({'logits': logits.detach().cpu()})

        return logits, fmap_list, masks, trace_outputs

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x shape: (B, C, H, W)
        B, C, H, W = x.size()
        Q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        K = self.key(x).view(B, -1, H * W)                     # (B, C//8, H*W)
        V = self.value(x).view(B, -1, H * W)                   # (B, C, H*W)

        attn_map = torch.bmm(Q, K)                             # (B, H*W, H*W)
        attn_weights = self.softmax(attn_map)
        out = torch.bmm(V, attn_weights.permute(0, 2, 1))      # (B, C, H*W)
        out = out.view(B, C, H, W)
        return out + x


class CrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim // 8)
        self.key = nn.Linear(in_dim, in_dim // 8)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context):  # x & context shape: (B, N, F)
        Q = self.query(x)
        K = self.key(context).transpose(1, 2)
        V = self.value(context)
        attn = torch.bmm(Q, K)
        attn_weights = self.softmax(attn)
        out = torch.bmm(attn_weights, V)
        return out + x


class GlobalAttention(nn.Module):
    def __init__(self, in_dim):
        super(GlobalAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim // 8)
        self.key = nn.Linear(in_dim, in_dim // 8)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x shape: (B, N, F)
        B, N, F = x.shape
        x_flat = x.view(B, N, F)
        global_q = self.query(x_flat.mean(dim=1, keepdim=True))  # (B, 1, d_k)
        global_k = self.key(x_flat).transpose(1, 2)              # (B, d_k, N)
        global_v = self.value(x_flat)                            # (B, N, F)

        attn = torch.bmm(global_q, global_k)                     # (B, 1, N)
        attn_weights = self.softmax(attn)                        # (B, 1, N)
        out = torch.bmm(attn_weights, global_v)                  # (B, 1, F)
        out = out.expand(-1, N, -1)                              # Broadcast to all slots
        return out + x
