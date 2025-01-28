# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CreditScoreRFM:
    """
    A class to calculate Recency, Frequency, Monetary values and perform RFM scoring with visualizations.
    """

    def __init__(self, rfm_data):
        self.rfm_data = rfm_data

    def calculate_rfm(self):
        self.rfm_data["TransactionStartTime"] = pd.to_datetime(
            self.rfm_data["TransactionStartTime"]
        )
        end_date = pd.Timestamp.utcnow()
        self.rfm_data["Last_Access_Date"] = self.rfm_data.groupby("CustomerId")[
            "TransactionStartTime"
        ].transform("max")
        self.rfm_data["Recency"] = (
            end_date - self.rfm_data["Last_Access_Date"]
        ).dt.days
        self.rfm_data["Frequency"] = self.rfm_data.groupby("CustomerId")[
            "TransactionId"
        ].transform("count")

        if "Amount" in self.rfm_data.columns:
            self.rfm_data["Monetary"] = self.rfm_data.groupby("CustomerId")[
                "Amount"
            ].transform("sum")
        else:
            self.rfm_data["Monetary"] = 1

        rfm_data = self.rfm_data[
            ["CustomerId", "Recency", "Frequency", "Monetary"]
        ].drop_duplicates()
        return rfm_data

    def calculate_rfm_scores(self, rfm_data):
        rfm_data["r_quartile"] = pd.qcut(
            rfm_data["Recency"], 4, labels=["4", "3", "2", "1"]
        )
        rfm_data["f_quartile"] = pd.qcut(
            rfm_data["Frequency"], 4, labels=["1", "2", "3", "4"]
        )
        rfm_data["m_quartile"] = pd.qcut(
            rfm_data["Monetary"], 4, labels=["1", "2", "3", "4"]
        )

        rfm_data["RFM_Score"] = (
            rfm_data["r_quartile"].astype(int) * 0.1
            + rfm_data["f_quartile"].astype(int) * 0.45
            + rfm_data["m_quartile"].astype(int) * 0.45
        )
        low_threshold = rfm_data["RFM_Score"].quantile(0.5)
        rfm_data["Risk_Label"] = rfm_data["RFM_Score"].apply(
            lambda x: "Good" if x >= low_threshold else "Bad"
        )
        return rfm_data

    def assign_label(self, rfm_data):
        low_threshold = rfm_data["RFM_Score"].quantile(0.5)
        rfm_data["Risk_Label"] = rfm_data["RFM_Score"].apply(
            lambda x: "Good" if x >= low_threshold else "Bad"
        )
        return rfm_data

    def plot_pairplot(self):
        sns.set_palette("pastel")
        sns.pairplot(
            self.rfm_data[["Recency", "Frequency", "Monetary"]], diag_kind="hist"
        )
        plt.suptitle("Pair Plot of RFM Variables", y=1.02)
        plt.show()

    def plot_heatmap(self):
        sns.set_palette("pastel")
        corr = self.rfm_data[["Recency", "Frequency", "Monetary"]].corr()
        sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
        plt.title("Correlation Matrix of RFM Variables")
        plt.show()

    def plot_histograms(self):
        sns.set_palette("pastel")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        sns.histplot(
            self.rfm_data["Recency"], bins=20, kde=True, ax=axes[0], color="skyblue"
        )
        axes[0].set_title("Recency Distribution")

        sns.histplot(
            self.rfm_data["Frequency"],
            bins=20,
            kde=True,
            ax=axes[1],
            color="lightgreen",
        )
        axes[1].set_title("Frequency Distribution")

        sns.histplot(
            self.rfm_data["Monetary"], bins=20, kde=True, ax=axes[2], color="lightcoral"
        )
        axes[2].set_title("Monetary Distribution")

        plt.tight_layout()
        plt.show()

    def calculate_counts(self, data):
        """
        Calculate good and bad counts for each RFM_bin.
        """
        grouped_data = data.groupby("RFM_bin")
        good_count = grouped_data["Risk_Label"].apply(lambda x: (x == "Good").sum())
        bad_count = grouped_data["Risk_Label"].apply(lambda x: (x == "Bad").sum())

        return good_count, bad_count

    def calculate_woe(self, good_count, bad_count):
        total_good = good_count.sum()
        total_bad = bad_count.sum()

        # Add epsilon (small value) to avoid log(0) or division by zero
        epsilon = 1e-10

        good_rate = good_count / (total_good + epsilon)  # Avoid division by zero
        bad_rate = bad_count / (total_bad + epsilon)  # Avoid division by zero

        # Calculate WoE
        woe = np.log(
            (good_rate + epsilon) / (bad_rate + epsilon)
        )  # Add epsilon to rates

        # Return WoE as a Series with the same index as good_count
        return pd.Series(woe, index=good_count.index)
