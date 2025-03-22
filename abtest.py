import pandas as pd
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu
from statsmodels.stats.multitest import multipletests

#因为样本不符合正态分布，所以用的是非参数检验
def perform_ab_test(group_a, group_b, label_a, label_b, target_column):

    t_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')

    print(f"\nA/B 测试 ({label_a} vs {label_b})")
    print(f"A 组 ({label_a}) 样本量: {len(group_a)}, B 组 ({label_b}) 样本量: {len(group_b)}")
    print(f"U-statistic: {t_stat:.3f}, p-value: {p_value:.10f}")

    if p_value < 0.05:
        print(f"{target_column} 在 A 组和 B 组之间存在显著差异！")
    else:
        print(f"A 组和 B 组的 {target_column} 无显著差异")

    return p_value


if __name__ == "__main__":

    df = pd.read_csv("data/data_process.csv")

    # 按视频类别分组
    theme_groups = df.groupby('video_category')[['like_type', 'relay_type']].mean()
    theme_groups.sort_values(by='like_type', ascending=False, inplace=True)

    theme_a = theme_groups.index[0]
    theme_b = theme_groups.index[-1]

    a_group = df[df['video_category'] == theme_a]['like_type']
    b_group = df[df['video_category'] == theme_b]['like_type']

    p_value1 = perform_ab_test(a_group, b_group, theme_a, theme_b, 'like_type')

    # 用户活跃度分组
    user_activity = df.groupby('user_id')[['total_action']].sum()
    median_activity = user_activity['total_action'].median()

    high_activity_users = user_activity[user_activity['total_action'] > median_activity].index
    low_activity_users = user_activity[user_activity['total_action'] <= median_activity].index

    a_group_user = df[df['user_id'].isin(high_activity_users)]['like_type']
    b_group_user = df[df['user_id'].isin(low_activity_users)]['like_type']

    p_value2 = perform_ab_test(a_group_user, b_group_user, '高活跃用户', '低活跃用户', 'like_type')