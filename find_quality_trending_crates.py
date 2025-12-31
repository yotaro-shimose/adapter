import polars as pl
from datetime import datetime, timezone


def main():
    base_dir = "data/createsio/2025-12-23-020016/data"
    crates_path = f"{base_dir}/crates.csv"
    downloads_path = f"{base_dir}/crate_downloads.csv"
    dependencies_path = f"{base_dir}/dependencies.csv"
    versions_path = f"{base_dir}/versions.csv"

    print("Loading dataframes...")
    df_crates = pl.read_csv(crates_path, ignore_errors=True)
    df_downloads = pl.read_csv(downloads_path)
    df_deps = pl.read_csv(dependencies_path, ignore_errors=True)
    df_versions = pl.read_csv(versions_path, ignore_errors=True)

    # Date parsing
    def parse_date(x):
        try:
            return datetime.fromisoformat(x)
        except (ValueError, TypeError, AttributeError):
            try:
                if isinstance(x, str):
                    return datetime.fromisoformat(x.replace("Z", "+00:00"))
            except Exception:
                return None
            return None

    print("Parsing dates...")
    df_crates = df_crates.with_columns(
        pl.col("created_at")
        .map_elements(parse_date, return_dtype=pl.Datetime("us", "UTC"))
        .alias("created_at")
    )

    # 1. Repository Age Filter
    print("Filtering old repositories...")
    repo_start_dates = (
        df_crates.lazy()
        .filter(pl.col("repository").is_not_null())
        .group_by("repository")
        .agg(pl.min("created_at").alias("repo_start_date"))
        .collect()
    )
    old_repo_cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
    new_repos = repo_start_dates.filter(pl.col("repo_start_date") >= old_repo_cutoff)

    # Filter crates to only those in new repos
    df_crates_new_repos = df_crates.join(new_repos, on="repository", how="inner")

    # 2. Dependency Analysis
    print("Analyzing dependencies...")
    # Map version_id -> crate_id
    version_to_crate = df_versions.select(["id", "crate_id"]).rename(
        {"id": "version_id", "crate_id": "dependant_id"}
    )

    # Enrich dependencies with dependant_id
    df_deps_enriched = df_deps.lazy().join(
        version_to_crate.lazy(), on="version_id", how="inner"
    )

    # Calculate direct dependant count (unique crates depending on this crate)
    direct_dependants = (
        df_deps_enriched.group_by("crate_id")
        .agg(pl.col("dependant_id").n_unique().alias("direct_dependant_count"))
        .collect()
    )

    # Identify "Frontend" crates:
    # 1. Join dependencies with repos for both sides
    repo_mapping = df_crates.select(["id", "repository"]).rename(
        {"id": "crate_id", "repository": "repo"}
    )

    internal_deps = (
        df_deps_enriched.join(
            repo_mapping.lazy(), on="crate_id", how="inner"
        )  # target crate repo
        .rename({"repo": "target_repo"})
        .join(
            repo_mapping.lazy(),
            left_on="dependant_id",
            right_on="crate_id",
            how="inner",
        )  # dependant crate repo
        .rename({"repo": "dependant_repo"})
        .filter(pl.col("target_repo") == pl.col("dependant_repo"))
        .select(["crate_id"])
        .unique()
        .collect()
    )

    # Crates that ARE dependencies within their own repo
    internal_dep_ids = internal_deps["crate_id"].to_list()

    # 3. Quality Filters & Frontend filtering
    print("Applying quality and frontend filters...")
    df_quality = df_crates_new_repos.filter(
        pl.col("description").str.len_chars() > 10,
        (pl.col("documentation").is_not_null()) & (pl.col("homepage").is_not_null()),
        pl.col("readme").str.len_chars() > 100,
        ~pl.col("id").is_in(
            internal_dep_ids
        ),  # Filter out internal dependencies -> Keep "Frontends"
    )

    # 4. Join with Downloads and Dependant counts
    print("Joining metrics...")
    q_joined = (
        df_quality.lazy()
        .join(df_downloads.lazy(), left_on="id", right_on="crate_id", how="left")
        .join(direct_dependants.lazy(), left_on="id", right_on="crate_id", how="left")
        .fill_null(0)
    )

    # 5. Result Selection and Sorting
    print("Finalizing results...")
    q_final = (
        q_joined.sort("direct_dependant_count", descending=True)
        .select(
            [
                "name",
                "created_at",
                "direct_dependant_count",
                "downloads",
                "description",
                "repository",
                "homepage",
                "documentation",
            ]
        )
        .head(20)
    )

    print("--- Top 20 Quality Frontend New Crates (Sorted by Direct Dependants) ---")
    with pl.Config(fmt_str_lengths=100, tbl_rows=25):
        print(q_final.collect())


if __name__ == "__main__":
    main()
