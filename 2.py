import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import math

class CustomLinear(nn.Module):
    """Custom linear layer with manual weight initialization"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
            
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class CustomAttention(nn.Module):
    """Pure PyTorch attention without cuDNN dependencies"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = CustomLinear(d_model, d_model)
        self.w_k = CustomLinear(d_model, d_model)
        self.w_v = CustomLinear(d_model, d_model)
        self.w_o = CustomLinear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(output)

class CustomTransformerLayer(nn.Module):
    """Custom transformer layer"""
    def __init__(self, d_model, n_heads, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = CustomAttention(d_model, n_heads, dropout)
        self.linear1 = CustomLinear(d_model, dim_feedforward)
        self.linear2 = CustomLinear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class CustomTransformer(nn.Module):
    """Pure PyTorch transformer"""
    def __init__(self, vocab_size=5000, d_model=256, n_heads=8, 
                 num_layers=4, dim_feedforward=512, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        self.layers = nn.ModuleList([
            CustomTransformerLayer(d_model, n_heads, dim_feedforward)
            for _ in range(num_layers)
        ])
        
        self.output_projection = CustomLinear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding)
        
    def forward(self, src_tokens, tgt_tokens):
        batch_size, src_len = src_tokens.size()
        _, tgt_len = tgt_tokens.size()
        
        src_emb = self.token_embedding(src_tokens) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pos_embedding[:, :src_len, :]
        src_emb = self.dropout(src_emb)
        
        tgt_emb = self.token_embedding(tgt_tokens) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pos_embedding[:, :tgt_len, :]
        tgt_emb = self.dropout(tgt_emb)
        
        for layer in self.layers:
            tgt_emb = layer(tgt_emb)
            
        return self.output_projection(tgt_emb)

class SemanticStateSpace(nn.Module):
    """Enhanced semantic space with better embeddings"""
    def __init__(self, vocab_size=5000, embedding_dim=256, hidden_dim=384):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.semantic_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.energy_network = nn.Sequential(
            CustomLinear(embedding_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            CustomLinear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            CustomLinear(hidden_dim // 4, 1)
        )
        
        nn.init.xavier_uniform_(self.semantic_embeddings.weight)
        
    def forward(self, token_ids):
        embeddings = self.semantic_embeddings(token_ids)
        sequence_embedding = embeddings.mean(dim=1)
        energy = self.energy_network(sequence_embedding)
        return energy, embeddings

class CreativityStateRegulator(nn.Module):
    """Fixed creativity regulator with proper mode switching"""
    def __init__(self):
        super().__init__()
        
        # Trainable parameters with different initial values
        self.lambda_base = nn.Parameter(torch.tensor(1.0))  # Higher base for more effect
        self.gamma_base = nn.Parameter(torch.tensor(1.0))
        
        # Mode-specific adjustments
        self.mode_parameters = {
            "strict_factual": {"lambda_scale": 0.1, "gamma_scale": 3.0},
            "balanced": {"lambda_scale": 1.0, "gamma_scale": 1.0},
            "creative": {"lambda_scale": 2.0, "gamma_scale": 0.5},
        }
        
        self.current_mode = "balanced"
        
    def set_mode(self, mode_name):
        self.current_mode = mode_name
        
    def get_creativity_parameters(self):
        params = self.mode_parameters.get(self.current_mode, self.mode_parameters["balanced"])
        lambda_t = self.lambda_base * params["lambda_scale"]
        gamma_t = self.gamma_base * params["gamma_scale"]
        
        return (torch.clamp(lambda_t, 0.05, 3.0), 
                torch.clamp(gamma_t, 0.1, 4.0))

class GSECompleteSystem(nn.Module):
    """Enhanced GSE System with Fixed Creativity Control"""
    def __init__(self, vocab_size=5000, d_model=256, n_heads=8, num_layers=4):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Core components
        self.transformer = CustomTransformer(vocab_size, d_model, n_heads, num_layers)
        self.semantic_space = SemanticStateSpace(vocab_size, d_model)
        self.creativity_regulator = CreativityStateRegulator()
        
    def set_creativity_mode(self, mode_name):
        self.creativity_regulator.set_mode(mode_name)
                
    def forward(self, src_tokens, tgt_tokens, task_type="balanced"):
        # Set mode based on task type
        self.set_creativity_mode(task_type)
        lambda_t, gamma_t = self.creativity_regulator.get_creativity_parameters()
        
        base_logits = self.transformer(src_tokens, tgt_tokens)
        modulated_logits = self._apply_gse_modulation(
            base_logits, src_tokens, lambda_t, gamma_t)
        
        return modulated_logits
    
    def _apply_gse_modulation(self, base_logits, src_tokens, lambda_t, gamma_t):
        batch_size, seq_len, vocab_size = base_logits.shape
        
        with torch.no_grad():
            _, context_embs = self.semantic_space(src_tokens)
            context_mean = context_embs.mean(dim=1, keepdim=True)
        
        all_embs = self.semantic_space.semantic_embeddings.weight
        
        # Enhanced similarity calculation with temperature
        context_norm = F.normalize(context_mean, p=2, dim=-1)
        all_embs_norm = F.normalize(all_embs, p=2, dim=-1)
        
        similarities = torch.matmul(context_norm, all_embs_norm.transpose(0, 1))
        
        # Apply temperature scaling based on creativity
        temperature = 0.3 + (lambda_t * 0.2)  # More creative = higher temperature
        similarities = F.softmax(similarities / temperature, dim=-1)
        
        novelty_scores = 1.0 - similarities
        coherence_scores = similarities
        
        # Expand for sequence
        novelty_scores = novelty_scores.expand(batch_size, seq_len, vocab_size)
        coherence_scores = coherence_scores.expand(batch_size, seq_len, vocab_size)
        
        # Stronger modulation effect
        gse_potential = -lambda_t * novelty_scores + gamma_t * coherence_scores
        
        return base_logits + (gse_potential * 2.0)  # Increased effect
    
    def generate(self, prompt, max_length=50, task_type="balanced", temperature=0.8):
        self.eval()
        generated = prompt.clone()
        
        # Set creativity mode for generation
        self.set_creativity_mode(task_type)
        lambda_t, gamma_t = self.creativity_regulator.get_creativity_parameters()
        
        # Adjust temperature based on creativity
        gen_temperature = temperature + (lambda_t.item() * 0.2)
        
        with torch.no_grad():
            for step in range(max_length):
                if generated.size(1) > 1:
                    tgt_input = generated[:, :-1]
                else:
                    tgt_input = generated
                
                logits = self.forward(prompt, tgt_input, task_type)
                next_token_logits = logits[:, -1, :] / gen_temperature
                
                # Apply top-k sampling for better quality
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == 2:  # EOS
                    break
                    
        return generated

class MemoryEfficientTrainer:
    """Improved trainer with better learning rate"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        
    def training_step(self, batch, task_type="balanced"):
        src, tgt = batch
        src, tgt = src.to(self.device), tgt.to(self.device)
        
        self.optimizer.zero_grad()
        
        output = self.model(src, tgt[:, :-1], task_type)
        loss = F.cross_entropy(
            output.reshape(-1, output.size(-1)), 
            tgt[:, 1:].reshape(-1),
            ignore_index=0
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

class GSEOrchestrator:
    """Enhanced orchestrator with better vocabulary"""
    def __init__(self, vocab_size=5000):
        #if torch.cuda.is_available():
        #    self.device = torch.device('cuda')
        #    print("Using GPU for computation")
        #else:
        self.device = torch.device('cuda')
        print("Using CPU for computation")
        
        print(f"Initializing GSE System on: {self.device}")
        
        self.model = GSECompleteSystem(vocab_size=vocab_size).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model Parameters: {total_params:,}")
        
        self.trainer = MemoryEfficientTrainer(self.model, self.device)
        
        # Enhanced vocabulary with actual words
        self.vocab = self._create_enhanced_vocab(vocab_size)
        
    def _create_enhanced_vocab(self, vocab_size):
        """Create a more meaningful vocabulary"""
        vocab = {
            0: "[PAD]", 1: "[UNK]", 2: "[EOS]", 3: "[BOS]",
            4: "the", 5: "of", 6: "and", 7: "to", 8: "a", 9: "in",
            10: "is", 11: "for", 12: "that", 13: "with", 14: "on",
            15: "as", 16: "by", 17: "this", 18: "are", 19: "from",
            20: "ai", 21: "intelligence", 22: "artificial", 23: "future",
            24: "creative", 25: "thinking", 26: "science", 27: "discovery",
            28: "meaning", 29: "story", 30: "about", 31: "system",
            32: "learning", 33: "machine", 34: "human", 35: "brain",
            36: "consciousness", 37: "semantic", 38: "exploration",
            39: "generative", 40: "model", 41: "neural", 42: "network",
            43: "algorithm", 44: "data", 45: "pattern", 46: "recognition",
            47: "innovation", 48: "imagination", 49: "creativity",
        }
        
        # Fill remaining with generic words
        for i in range(50, min(1000, vocab_size)):
            vocab[i] = f"concept_{i}"
            
        for i in range(1000, vocab_size):
            vocab[i] = f"token_{i}"
            
        return vocab
    
    def train(self, dataloader, epochs=2, task_type="balanced"):
        print(f"🧠 Training for {epochs} epochs in {task_type} mode...")
        
        for epoch in range(epochs):
            total_loss, num_batches = 0, 0
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    loss = self.trainer.training_step(batch, task_type)
                    total_loss += loss
                    num_batches += 1
                    
                    if batch_idx % 5 == 0:
                        print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}")
                        
                except Exception as e:
                    print(f"Batch {batch_idx} error: {e}")
                    continue
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"✅ Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
    
    def generate_text(self, prompt, max_length=20, creativity_mode="balanced"):
        prompt_tokens = self._text_to_tokens(prompt).unsqueeze(0).to(self.device)
        
        generated_tokens = self.model.generate(
            prompt_tokens, 
            max_length=max_length,
            task_type=creativity_mode
        )
        
        return self._tokens_to_text(generated_tokens[0])
    
    def _text_to_tokens(self, text):
        words = text.lower().split()
        tokens = [3]  # BOS
        
        for word in words:
            # Try to find word in vocabulary
            token_id = None
            for vid, vword in self.vocab.items():
                if vword.lower() == word:
                    token_id = vid
                    break
            
            if token_id is None:
                # Fallback to hash-based assignment
                token_id = hash(word) % (self.model.vocab_size - 100) + 100
                
            tokens.append(token_id)
            
        tokens.append(2)  # EOS
        return torch.tensor(tokens, dtype=torch.long)
    
    def _tokens_to_text(self, tokens):
        token_list = tokens.cpu().tolist()
        words = []
        for token_id in token_list:
            if token_id in self.vocab:
                words.append(self.vocab[token_id])
            elif token_id == 0:
                continue
            else:
                words.append(f"[{token_id}]")
        return " ".join(words)
    
    def demo_creativity_spectrum(self):
        """Enhanced demonstration with proper parameter display"""
        test_prompts = [
            "the future of",
            "artificial intelligence",
            "creative thinking"
        ]
        
        print("\n" + "="*70)
        print("🎨 CREATIVITY SPECTRUM DEMONSTRATION")
        print("="*70)
        
        for prompt in test_prompts:
            print(f"\n📝 Prompt: '{prompt}'")
            for mode in ["strict_factual", "balanced", "creative"]:
                # Get parameters first
                self.model.set_creativity_mode(mode)
                lambda_t, gamma_t = self.model.creativity_regulator.get_creativity_parameters()
                
                result = self.generate_text(prompt, creativity_mode=mode)
                print(f"  {mode:15} | λ={lambda_t.item():.2f} γ={gamma_t.item():.2f}")
                print(f"                 → '{result}'")

def create_meaningful_data(vocab_size=5000, num_samples=200):
    """Create more meaningful training data"""
    data = []
    
    # Common patterns for the model to learn
    patterns = [
        [4, 23, 5, 22, 21],  # "the future of artificial intelligence"
        [22, 21, 10, 24],     # "artificial intelligence is creative"
        [24, 25, 10, 47],     # "creative thinking is innovation"
        [26, 27, 10, 32],     # "science discovery is learning"
        [4, 28, 5, 36],       # "the meaning of consciousness"
    ]
    
    for i in range(num_samples):
        # Use patterns with variations
        pattern = patterns[i % len(patterns)]
        src = torch.tensor(pattern, dtype=torch.long)
        
        # Create target with some variation
        tgt = src.clone()
        if len(tgt) > 2:
            # Sometimes modify one token
            if torch.rand(1) > 0.7:
                mod_idx = torch.randint(1, len(tgt)-1, (1,)).item()
                tgt[mod_idx] = torch.randint(10, min(100, vocab_size), (1,))
        
        data.append((src, tgt))
    
    return data

class SimpleDataLoader:
    def __init__(self, data, batch_size=4):
        self.data = data
        self.batch_size = batch_size
        
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx >= len(self.data):
            raise StopIteration
            
        batch = self.data[self.idx:self.idx + self.batch_size]
        self.idx += self.batch_size
        
        src_seqs = [item[0] for item in batch]
        tgt_seqs = [item[1] for item in batch]
        
        src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
        
        return src_padded, tgt_padded

def main():
    print("🎯 ENHANCED GSE SYSTEM - Optimized for T2000 Quadro")
    print("✨ With Fixed Creativity Control and Better Vocabulary")
    print("="*70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    orchestrator = GSEOrchestrator(vocab_size=5000)
    
    print("Creating meaningful training data...")
    train_data = create_meaningful_data(num_samples=100)
    train_loader = SimpleDataLoader(train_data, batch_size=4)
    
    print("Starting training...")
    orchestrator.train(train_loader, epochs=2, task_type="balanced")
    
    print("\nDemonstrating enhanced creativity control...")
    orchestrator.demo_creativity_spectrum()
    
    print("\n" + "="*70)
    print("🚀 INTERACTIVE GENERATION WITH PROPER CREATIVITY CONTROL")
    print("="*70)
    
    examples = [
        ("the future of", "strict_factual"),
        ("the future of", "creative"),
        ("ai and human", "balanced"),
        ("neural network", "creative"),
    ]
    
    for prompt, mode in examples:
        result = orchestrator.generate_text(prompt, creativity_mode=mode)
        # Get current parameters
        orchestrator.model.set_creativity_mode(mode)
        lambda_t, gamma_t = orchestrator.model.creativity_regulator.get_creativity_parameters()
        print(f"🎯 {mode:15} | λ={lambda_t.item():.2f} γ={gamma_t.item():.2f}")
        print(f"   '{prompt}' → '{result}'\n")
    
    print("✅ GSE System Ready with Proper Creativity Control!")

if __name__ == "__main__":
    main()