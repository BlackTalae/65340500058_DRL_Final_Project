from scipy.stats import t, norm
import numpy as np
import pickle

def t_test(group1, group2, alpha=0.05, tail="two-sided", verbose=True, name=""):
    """
    Welch's t-test แบบกำหนด one-tailed หรือ two-tailed
    Parameters:
        group1, group2 : list or np.array
        alpha : ระดับนัยสำคัญ
        tail : "two-sided", "greater", "less"
        verbose : แสดงผลลัพธ์หรือไม่
    """

    group1 = np.array(group1)
    group2 = np.array(group2)

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    t_stat = (mean1 - mean2) / np.sqrt((var1 / n1) + (var2 / n2))

    df_numerator = (var1 / n1 + var2 / n2) ** 2
    df_denominator = ((var1 / n1)**2) / (n1 - 1) + ((var2 / n2)**2) / (n2 - 1)
    df = df_numerator / df_denominator

    # === เลือกทิศทาง ===
    if tail == "two-sided":
        p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    elif tail == "greater":  # H1: group1 > group2
        p_value = 1 - t.cdf(t_stat, df)
    elif tail == "less":     # H1: group1 < group2
        p_value = t.cdf(t_stat, df)
    else:
        raise ValueError("Invalid tail value. Choose 'two-sided', 'greater', or 'less'.")

    if verbose:
        print("\n📊 Welch's t-test")
        print(f"PPO mean = {mean1:.2f}, n = {n1}")
        print(f"SAC mean = {mean2:.2f}, n = {n2}")
        print(f"t-statistic = {t_stat:.4f}")
        print(f"df ≈ {df:.2f}")
        print(f"p-value = {p_value:.4f} ({tail})")

        if p_value < alpha:
            print(f"❌ ปฏิเสธ H₀ (p < {alpha}) → มีนัยสำคัญ")
        else:
            print(f"✅ ยอมรับ H₀ (p ≥ {alpha}) → ไม่มีนัยสำคัญ")

    return t_stat, p_value, df

def z_test(group1, group2, alpha=0.05, tail="two-sided", verbose=True, name=""):
    """
    Z-test สำหรับการเปรียบเทียบค่าเฉลี่ยของกลุ่มตัวอย่างขนาดใหญ่ (n ≥ 30)
    
    Parameters:
        group1, group2 : list หรือ np.array
        alpha : ค่าระดับนัยสำคัญ (default 0.05)
        tail : "two-sided", "greater", หรือ "less"
        verbose : แสดงผลลัพธ์หรือไม่

    Returns:
        z_stat, p_value
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # z-statistic
    z_stat = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))

    # p-value ตาม tail
    if tail == "two-sided":
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    elif tail == "greater":  # H1: mean1 > mean2
        p_value = 1 - norm.cdf(z_stat)
    elif tail == "less":     # H1: mean1 < mean2
        p_value = norm.cdf(z_stat)
    else:
        raise ValueError("tail ต้องเป็น 'two-sided', 'greater', หรือ 'less'")

    if verbose:
        print(f"\n📊 Z-test for Difference in Means of {name}")
        print(f"PPO: mean = {mean1:.2f}, std = {std1:.2f}, n = {n1}")
        print(f"SAC: mean = {mean2:.2f}, std = {std2:.2f}, n = {n2}")
        print(f"z-statistic = {z_stat:.4f}")
        print(f"p-value     = {p_value:.4f} ({tail})")
        if p_value < alpha:
            print(f"❌ ปฏิเสธ H₀: มีนัยสำคัญ (p < {alpha})")
        else:
            print(f"✅ ยอมรับ H₀: ยังไม่มีหลักฐานว่าค่าเฉลี่ยต่างกัน (p ≥ {alpha})")

    return z_stat, p_value

with open("PPO_testing_data.pkl", "rb") as f:
    PPO_testing_data = pickle.load(f)

with open("SAC_testing_data.pkl", "rb") as f:
    SAC_testing_data = pickle.load(f)

from scipy.stats import norm
import numpy as np

def z_proportion_test(success1, n1, success2, n2, alpha=0.05, tail="two-sided", verbose=True, name=""):
    """
    Z-test สำหรับเปรียบเทียบสัดส่วนของสองกลุ่ม (เช่น success rate)
    
    Parameters:
        success1 : จำนวนความสำเร็จของกลุ่มที่ 1
        n1 : จำนวนทั้งหมดในกลุ่มที่ 1
        success2 : จำนวนความสำเร็จของกลุ่มที่ 2
        n2 : จำนวนทั้งหมดในกลุ่มที่ 2
        alpha : ค่าระดับนัยสำคัญ
        tail : "two-sided", "greater", หรือ "less"
        verbose : แสดงผลหรือไม่

    Returns:
        z_stat, p_value
    """
    p1 = success1 / n1
    p2 = success2 / n2
    p_pool = (success1 + success2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    z_stat = (p1 - p2) / se

    if tail == "two-sided":
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    elif tail == "greater":
        p_value = 1 - norm.cdf(z_stat)
    elif tail == "less":
        p_value = norm.cdf(z_stat)
    else:
        raise ValueError("tail ต้องเป็น 'two-sided', 'greater', หรือ 'less'")

    if verbose:
        print("\n📊 Z-test for Difference in Proportions")
        print(f"PPO: {success1}/{n1} = {p1:.2%}")
        print(f"SAC: {success2}/{n2} = {p2:.2%}")
        print(f"z-statistic = {z_stat:.4f}")
        print(f"p-value     = {p_value:.4f} ({tail})")
        if p_value < alpha:
            print(f"❌ ปฏิเสธ H₀: มีนัยสำคัญ (p < {alpha})")
        else:
            print(f"✅ ยอมรับ H₀: ยังไม่มีหลักฐานว่าสัดส่วนต่างกัน (p ≥ {alpha})")

    return z_stat, p_value

z_test(PPO_testing_data["entropies"], SAC_testing_data["entropies"], tail="less", name="Spectral Flatness")
z_test(PPO_testing_data["action_mse"], SAC_testing_data["action_mse"], tail="less", name="Action MSE")
z_proportion_test(np.sum(PPO_testing_data["successes"]), len(PPO_testing_data["successes"]), np.sum(SAC_testing_data["successes"]), len(SAC_testing_data["successes"]), tail="less", name="Spectral Flatness")
z_test(PPO_testing_data["rewards"], SAC_testing_data["rewards"], tail="less", name="Spectral Flatness")
t_test(PPO_testing_data["reward_var"], SAC_testing_data["reward_var"], tail="greater", name="Reward Variance")