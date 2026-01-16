import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_potts_binder(q_value, lattices, L_list):
    plt.figure(figsize=(10, 7))
    
    # 色のリスト
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    for lattice in lattices:
        for i, L in enumerate(L_list):
            # ファイル名の構成: potts_square_q4_L32.csv など
            filename = f"potts_{lattice}_q{q_value}_L{L}.csv"
            
            if not os.path.exists(filename):
                print(f"Warning: {filename} not found. Skipping...")
                continue
                
            # データの読み込み
            df = pd.read_csv(filename)
            
            # プロット (格子形状とLをラベルに含める)
            label = f"{lattice} (L={L})"
            plt.plot(df['beta'], df['binder_cumulant'], marker='o', markersize=3, label=label)

        # 理論的な転移点 beta_c の描画
        # 正方格子の場合: beta_c = ln(1 + sqrt(q))
        if lattice == "square":
            beta_c = np.log(1 + np.sqrt(q_value))
            plt.axvline(x=beta_c, color='black', linestyle='--', alpha=0.5, 
                        label=f"Theory beta_c (Square): {beta_c:.4f}")
        
        elif lattice == "honeycomb" or lattice == "triangular":
            coeffs = [1, 3, 0, -q_value]
            y_c = [r.real for r in np.roots(coeffs) if np.isreal(r) and r > 0][0]
            if lattice == "honeycomb":
                beta_c = np.log(1.0 + y_c)
            else: # triangular
                beta_c = np.log(1.0 + q_value / y_c)

    plt.title(f"{q_value}-state Potts Model Binder Cumulant")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$U_4 = \langle m^4 \rangle / \langle m^2 \rangle^2$")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # 保存
    plt.savefig(f"potts_q{q_value}_binder_plot.png", dpi=300)
    plt.show()

# --- 設定 ---
Q_VALUE = 4               # 描画したいqの値
LATTICES = ["honeycomb"]     # 格子形状 ["square", "triangular", "honeycomb"]
L_SIZES = [4, 8, 16, 32, 64, 128]    # 読み込むLのリスト

# 実行
if __name__ == "__main__":
    plot_potts_binder(Q_VALUE, LATTICES, L_SIZES)