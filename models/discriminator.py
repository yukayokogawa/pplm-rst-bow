import torch
import torch.nn as nn
from .classification_head import ClassificationHead
from transformers import BartConfig, BartForConditionalGeneration
from torch_scatter import scatter_mean

class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
        self,
        class_size=2
    ):
        super(Discriminator, self).__init__()
        
        self.latent_hidden_size = 256
        config = BartConfig.from_pretrained('facebook/bart-base')
        self.model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-base',
            config=config,
        )
        self.encoder = self.model.model.decoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.config = config
        
        self.proj = nn.Linear(self.config.d_model, self.latent_hidden_size)
        self.classifier_head = ClassificationHead(
            class_size=class_size,
            embed_size=self.latent_hidden_size,
        )

    def get_classifier(self):
        return self.classifier_head

    
    def forward(self, input_ids, discourse_pos=None):
        #print(discourse_pos)
        outputs = self.encoder(input_ids=input_ids)
        hidden = outputs[0]
        disc_rep = scatter_mean(hidden, discourse_pos.unsqueeze(-1).expand(-1, discourse_pos.size(1), hidden.size(2)), dim=1)
        head_rep = self.proj(disc_rep[:,:-1]).contiguous()
        tail_rep = self.proj(disc_rep[:,1:]).contiguous()
        rel_logits = self.classifier_head(head_rep, tail_rep)

        return rel_logits

    def predict(self, input_sentence):
        input_t = self.tokenizer.encode(input_sentence)
        input_t = torch.tensor([input_t], dtype=torch.long, device=self.device)
        if self.cached_mode:
            input_t = self.avg_representation(input_t)

        log_probs = self(input_t).data.cpu().numpy().flatten().tolist()
        prob = [math.exp(log_prob) for log_prob in log_probs]
        return prob
