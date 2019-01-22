1. 如果 H' 包含於 H 的話，一般來說，因為 H' 更不容易 capture 到 f 的樣子所以 deterministic noise 更大
2. H(10, 3) \belongs H(10, 4)
3. w_{t+1} = (1 - 2\eta\lambda / N)w_{t} - \eta \nabla E_in
4. 
w_reg(\lambda) is always a constant function of \lambda for \lambda >= 0: 不對當 lambda 小到一定程度之後， w_reg(lambda) 就會永遠都是 w_lin 了
w_reg is non decreasing function of lambda: 不對，當 lambda 越大 ||w_reg|| 會被壓越小, 是 non increasing function
||w_reg|| <= || w_lin || : 對，因為當 lambda 越大 ||w_reg||會被壓越小

5. 選 sqrt(9 + 4 sqrt(6))
6. 總共 5 周, 所以最少要 2^5 = 32 人
投出第一週之後，剩下 2~5 週需要至少 16 人（因為已經爆了一半的人了）
7. Cost: (32 + 16 + 8 + 4 + 2 ) * 10 + 10 = 630
淨賺: 1000 - 630 = 370
8. 1 因為你一開始就導好公式了
9. 用 multi-bin hoeffding
P(|E_out(g) - E_in(g)=0 | > 1/100) <= 2 * exp(-2 * (1/100)^2 * 10000) = 0.271
10. a(x) AND g(x): 因為你只接受你覺得 g 有信心的, 這時候一定會被 Hoeffding bound 住(還有原因是你當初的 data 其實是從 a(x) 過濾出來的)
11. (X^T X + X^{\tilde} X^{\tilde})^{-1} (X^{T} y + X^{\tilde}^T y\tilde)
12. X\tilde = sqrt(lambda) I, y\tilde = 0
13. E_in = 0.05, E_out = 0.045
14. lambda = 1e-8, E_in = 0.015, E_out = 0.02
15. log lambda = -7
16. log lambda = -8
17. log lambda = 0
18. E_in = 0.035, E_out = 0.020
19. log = -8
20. E_in = 0.015, E_out = 0.020

