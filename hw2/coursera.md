1. lambda mu + (1- lambda) (1- mu)
2. 0.5
3. 460000
4. ovb = 0.63217, rpb = 0.331308, pvb = 0.223, d = 0.215, vvb = 0.86 => devoye
5. ovb = 13.828, rpb = 7.048, pvb = 5.10, d = 5.59, vvb = 16.26 => pvb
6. 
pf: 
(選 k+1 不重複的 2 個點中間放 +1 or -1)  - (有選到邊際 2 個中的 1個區間 * 中間的 k-1區間選 1個點)
2 * ((k+1) choose 2) - 2 * (k-1) = k^2 + k - 2k -2 = k^2 -k +2 
7. 最多能 shatter 3 個點
8. 最多能 shatter 的資料形式就是 data 排在半徑的射線上，所以是: N+1 choose 2 + 1
9. 如果是 D 維度的多項式，在數線上最多有 D 個搖桿，只要能 shatter 出 oxoxoxo 交錯的，其他都 shatter 得出來，因為一旦 shatter 的出 oxoxoxo的形式，任何有連續 oo 或 xx 的都不需要這麼多搖桿。因此給定 D 維，也就是 D 個搖桿，最多能 shatter 出 D+1 個資料使得這個資料是 oxoxoxo 的形式
10. d 維最多能開出 2^d 個 rectangle, 又因為 S 可以任意指定那些 rectangle 是 +1, 所以 2^d 必定 shatter 得出來。又 2^d+1 個點的話，怎麼擺都會有至少 2 個點在同一個 rectangle, 所以就分不開
11. 
12. 
m_H(floor(N/2)): 反例: positive intervals, m_H(3) = 7, m_H(1) = 2
2^d_vc: 反例: positive intervals, m_H(3) = 7, 2^d_vc = 2^2 = 4
數學歸納法: 
    Base case: N = d_vc 時, min{...} - m_H(d_vc) = 0 (因為 d_vc 就是最高能 shatter 的 data 數)
    假設 N = d_vc + k 時, 成立
    N = d_vc + k + 1 時, 把 induction hypothesis 帶入之後可以得到 2*m_H(d_vc+k) - m_H(d_vc + k + 1)
    我們知道這個式子會大於 0, 因為當知道 d_vc+k 最多能產生出 m_H(d_vc+k) 種 dichotomy, 所以再多加一筆資料後，假設這筆資料放了 o/x 之後， 剩下 d_vc+k 筆資料也最多只能產生出 m_H(d_vc+k) 種 dichotomy, 因此得到 2*m_H(d_vc+k) >= m_H(d_vc + k + 1)
    得證  
N^(d_k/2): 反例: positive intervals, m_H(3) = 7, 3^(d_vc=2/2) = 3


13. 
2^N: 可以, 就完全 shatter
2^floor(sqrt(N)): 此 growth function breakpoint 在 k = 2, 但卻在夠大的 N 時, 不能被 bounding function 壓住 -> 不存在此 growth function
1: 可以, 假設 H = { h | h(x) = 1} 
N^2 - N + 2: 上面就有這個例子

14. 
直觀來說: 交集的 d_vc 應該要比任何一個 d_vc(H_k) 都小

15. 
https://github.com/beader/mlnotebook/blob/master/section2/vc-dimension-three.md

16. 把試子爆開
s = +, theta >= 0: 0.3 theta + 0.2
s = +, theta <= 0: -0.3 theta + 0.2
s = -, theta >= 0: -0.3 theta + 0.8
s = -, theta <= 0: 0.3 theta + 0.8
答案: 0.5 + 0.3s(|theta| - 1)

17.

18.

19.

20.

