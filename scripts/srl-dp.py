
def srl_dp(b_score, e_score, n):
    res = [0]*n
    stat = {}
    stat['B'] = [0]
    stat['E'] = [1]
    dp = []
    dp.append(b_score[0])
    dp.append(e_score[0])
    for i in range(1,n):
        b_s = [dp[0], b_score[i], dp[0]+b_score[i], dp[1]+b_score[i]]
        e_s = [dp[0]+e_score[i], dp[1]]
        b_ms = max(b_s)
        stb = b_s.index(b_ms)
        stat['B'].append(stb)
        e_ms = max(e_s)
        ste = e_s.index(e_ms)
        stat['E'].append(ste)
        dp[0] = b_ms
        dp[1] = e_ms

    f = [dp[0], dp[1], 0]
    max_score = max(f)
    max_index = f.index(max_score)
    next = ''
    if max_index == 2:
        return res
    elif max_index == 0:
        init = stat['B'][-1]
        if init == 0:
            next = 'B'
        elif init == 1:
            res[-1] = 1
            return res
        elif init == 2:
            next = 'B'
            res[-1] = 1
        else:
            next = 'E'
            res[-1] = 1
    else:
        init = stat['E'][-1]
        if init == 0:
            next = 'B'
            res[-1] = 2
        else:
            next = 'E'

    print(stat)
    for i in range(1, n):
        if n-1-i == 0:
            if next == 'B':
                res[0] = 1
            else:
                res[0] = 2
            return res
        if next == 'B':
            st = stat['B'][n-1-i]
            if st == 0:
                next = 'B'
            elif st == 1:
                res[n-1-i] = 1
                return res
            elif st == 2:
                next = 'B'
                res[n-1-i] = 1
            else:
                next = 'E'
                res[n-1-i] = 1
        else:
            st = stat['E'][n-1-i]
            if st == 0:
                next = 'B'
                res[n-1-i] = 2
            else:
                next = 'E'

    return res

b_s = [10,1,-3,0,-3,-7,-9]
e_s = [-5,-7,-6,-3,-4,9,1]
res = srl_dp(b_s,e_s,7)
print(res)

