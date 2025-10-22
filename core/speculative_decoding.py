"""
Speculative Decoding Implementation
Uses TinyLlama as draft model with MobileVLM-V2 as verifier
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from llama_cpp import Llama

from core.inference_engine import InferenceConfig
from core.metrics import SpeculativeMetrics


class SpeculativeDecoder:
    """
    Implements speculative decoding for faster inference
    Draft model generates K tokens, verifier accepts or rejects
    """
    
    def __init__(
        self,
        verifier_model_path: str,
        draft_model_path: str,
        draft_tokens: int = 4,
        acceptance_threshold: float = 0.85,
        n_threads: int = 4,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.draft_tokens = draft_tokens
        self.acceptance_threshold = acceptance_threshold
        self.metrics = SpeculativeMetrics()
        
        # Load verifier model (MobileVLM)
        self.logger.info(f"Loading verifier model: {verifier_model_path}")
        self.verifier = self._load_model(verifier_model_path, n_threads)
        
        # Load draft model (TinyLlama)
        self.logger.info(f"Loading draft model: {draft_model_path}")
        self.draft_model = self._load_model(draft_model_path, n_threads)
        
        self.logger.info("Speculative decoder initialized")
    
    def _load_model(self, model_path: str, n_threads: int) -> Llama:
        """Load a quantized model"""
        try:
            return Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=n_threads,
                n_batch=512,
                use_mmap=True,
                f16_kv=True,
                n_gpu_layers=0,  # CPU only
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using speculative decoding
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            repeat_penalty: Repetition penalty
            stop_sequences: Stop sequences
            
        Returns:
            Generated text with metrics
        """
        start_time = time.time()
        
        current_prompt = prompt
        generated_text = ""
        total_tokens = 0
        accepted_sequences = 0
        rejected_sequences = 0
        
        while total_tokens < max_tokens:
            # Step 1: Draft model generates K tokens
            draft_start = time.time()
            draft_output = self._draft_generate(
                current_prompt + generated_text,
                num_tokens=min(self.draft_tokens, max_tokens - total_tokens),
                temperature=temperature,
                top_p=top_p
            )
            draft_time = time.time() - draft_start
            
            # Step 2: Verifier checks draft tokens
            verify_start = time.time()
            verification_result = self._verify_tokens(
                current_prompt + generated_text,
                draft_output['tokens'],
                temperature=temperature,
                top_p=top_p
            )
            verify_time = time.time() - verify_start
            
            # Step 3: Accept or reject
            if verification_result['accepted']:
                # Accept draft tokens
                accepted_text = draft_output['text']
                generated_text += accepted_text
                total_tokens += len(draft_output['tokens'])
                accepted_sequences += 1
                
                self.logger.debug(
                    f"Accepted {len(draft_output['tokens'])} draft tokens "
                    f"(confidence: {verification_result['confidence']:.3f})"
                )
            else:
                # Reject and use verifier's token
                verifier_token = verification_result['verifier_token']
                generated_text += verifier_token
                total_tokens += 1
                rejected_sequences += 1
                
                self.logger.debug(
                    f"Rejected draft, using verifier token "
                    f"(confidence: {verification_result['confidence']:.3f})"
                )
            
            # Check stop sequences
            if stop_sequences and any(seq in generated_text for seq in stop_sequences):
                break
        
        total_time = time.time() - start_time
        
        # Calculate acceptance rate
        total_sequences = accepted_sequences + rejected_sequences
        acceptance_rate = accepted_sequences / total_sequences if total_sequences > 0 else 0
        
        # Record metrics
        self.metrics.record_inference(
            total_tokens=total_tokens,
            accepted_tokens=accepted_sequences * self.draft_tokens,
            acceptance_rate=acceptance_rate,
            speedup=total_tokens / total_time if total_time > 0 else 0
        )
        
        return {
            'text': generated_text,
            'tokens_generated': total_tokens,
            'latency': total_time,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            'acceptance_rate': acceptance_rate,
            'accepted_sequences': accepted_sequences,
            'rejected_sequences': rejected_sequences,
            'speedup_factor': self._estimate_speedup(acceptance_rate)
        }
    
    def _draft_generate(
        self,
        prompt: str,
        num_tokens: int,
        temperature: float,
        top_p: float
    ) -> Dict[str, Any]:
        """Draft model generates tokens quickly"""
        output = self.draft_model(
            prompt,
            max_tokens=num_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False
        )
        
        text = output['choices'][0]['text']
        # Approximate token count (exact tokenization may vary)
        tokens = text.split()
        
        return {
            'text': text,
            'tokens': tokens
        }
    
    def _verify_tokens(
        self,
        prompt: str,
        draft_tokens: List[str],
        temperature: float,
        top_p: float
    ) -> Dict[str, Any]:
        """
        Verifier checks if draft tokens are acceptable
        
        Returns verification result with acceptance decision
        """
        # Generate one pass with verifier to compare
        draft_text = ' '.join(draft_tokens)
        
        # Get verifier's prediction for the same sequence
        verifier_output = self.verifier(
            prompt,
            max_tokens=len(draft_tokens),
            temperature=temperature,
            top_p=top_p,
            echo=False,
            logprobs=1
        )
        
        verifier_text = verifier_output['choices'][0]['text']
        
        # Calculate similarity/confidence
        # Simple approach: check if draft matches verifier closely
        draft_lower = draft_text.lower().strip()
        verifier_lower = verifier_text.lower().strip()
        
        # Token-level comparison
        draft_tok = draft_lower.split()
        verifier_tok = verifier_lower.split()
        
        if not draft_tok or not verifier_tok:
            confidence = 0.0
        else:
            matching_tokens = sum(1 for d, v in zip(draft_tok, verifier_tok) if d == v)
            confidence = matching_tokens / max(len(draft_tok), len(verifier_tok))
        
        accepted = confidence >= self.acceptance_threshold
        
        return {
            'accepted': accepted,
            'confidence': confidence,
            'verifier_token': verifier_text.split()[0] if verifier_text else ""
        }
    
    def _estimate_speedup(self, acceptance_rate: float) -> float:
        """
        Estimate speedup factor from speculative decoding
        
        Theoretical speedup = (K * acceptance_rate + 1) / (1 + alpha)
        where K = draft tokens, alpha = draft overhead
        """
        K = self.draft_tokens
        alpha = 0.3  # Estimated draft model overhead (30% of verifier time)
        
        speedup = (K * acceptance_rate + 1) / (1 + alpha)
        return speedup
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get speculative decoding metrics"""
        return self.metrics.get_summary()
    
    def reset_cache(self):
        """Reset both model caches"""
        self.verifier.reset()
        self.draft_model.reset()
        self.logger.info("Speculative decoder caches cleared")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'verifier'):
            del self.verifier
        if hasattr(self, 'draft_model'):
            del self.draft_model

