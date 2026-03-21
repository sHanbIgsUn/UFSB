import numpy as np
from typing import Set

class ConformalPredictor:
    def __init__(self, k_reg: int = 10, lambda_val: float = 0.3, tau: float = 5.0):
        self.k_reg = k_reg          # 排名正则化超参数
        self.lambda_val = lambda_val  # 固定正则化系数（不再动态调整）
        self.tau = tau              # 置信度阈值
    
    def compute_rank(self, probs: np.ndarray) -> np.ndarray:
        sorted_indices = np.argsort(probs)[::-1]
        ranks = np.zeros_like(probs)
        for i, idx in enumerate(sorted_indices):
            ranks[idx] = i + 1
        return ranks
    
    def compute_cumulative_prob(self, probs: np.ndarray, rank: np.ndarray, target_label: int) -> float:
        higher_ranks = np.where(rank < rank[target_label])[0]
        return np.sum(probs[higher_ranks])
    
    def generate_prediction_set(self, g_probs: np.ndarray) -> Set[int]:
        probs=g_probs.cpu().detach().numpy()
        n_labels = len(probs)
        ranks = self.compute_rank(probs)
        prediction_set = set()
        
        for label in range(n_labels):
            pi_y = probs[label]
            rho_y = self.compute_cumulative_prob(probs, ranks, label)
            o_y = ranks[label]
            # 固定正则化惩罚项（原动态调整部分替换为固定lambda_val）
            penalty = self.lambda_val * max(0, o_y - self.k_reg)
            score = rho_y + pi_y + penalty
            if  score<= self.tau:
                prediction_set.add(label)

        pred_probs = [probs[y] for y in prediction_set]
        if not pred_probs:
            pred_span = 0.0
        else:
            pred_span = max(pred_probs) - min(pred_probs)
        return prediction_set,pred_span

if __name__ == "__main__":
    sample_probs = np.array([0.95, 0.80, 0.92, 0.50])
    true_label = 0  
    
    predictor = ConformalPredictor()
    pred_set = predictor.generate_prediction_set(probs=sample_probs)
    
    print(f"高置信度预测集: {pred_set}")
    print(f"预测集大小: {len(pred_set)}（越小表示不确定性越低）")