import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(df):

    video_theme_stats = df.groupby("video_category").agg(
        total_likes=("like_type", "sum"),
        total_videos=("like_type", "count"),
        avg_like_rate=("like_type", "mean"),
    ).sort_values(by="avg_like_rate", ascending=False)


    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

    sns.lineplot(ax=axes[0], x=video_theme_stats.index, y=video_theme_stats["total_videos"])
    axes[0].set_title("Total Videos for Different Video Themes", fontsize=14)
    axes[0].set_ylabel("Total Videos", fontsize=12)

    sns.lineplot(ax=axes[1], x=video_theme_stats.index, y=video_theme_stats["total_likes"])
    axes[1].set_title("Total Likes for Different Video Themes", fontsize=14)
    axes[1].set_ylabel("Total Likes", fontsize=12)

    sns.lineplot(ax=axes[2], x=video_theme_stats.index, y=video_theme_stats["avg_like_rate"])
    axes[2].set_title("Average Like Rate of Different Video Themes", fontsize=14)
    axes[2].set_ylabel("Average Like Rate", fontsize=12)

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")

    for ax in axes:
        ax.set_xticklabels(video_theme_stats.index, rotation=45, ha="right", fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    df = pd.read_csv("data/data_process.csv")

    plot_data(df)