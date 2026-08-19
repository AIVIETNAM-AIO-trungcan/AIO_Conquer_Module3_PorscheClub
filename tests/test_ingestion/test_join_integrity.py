"""Test toàn vẹn join giữa train/features (A2, A3, A4) — docs/05_test_plan.md mục 2.2.

A2/A3 test trên sample_train RAW (còn Dept) — đúng ý nghĩa ở bước ingestion,
TRƯỚC aggregate_to_store_date (xem docs/00_decisions.md [2026-08-19]
"Đổi đơn vị dự báo"). Sau aggregate, đơn vị là (Store, Date) — join với
features.csv trở thành 1-1 (không còn N-1 từ góc nhìn nhiều Dept dùng chung
1 dòng features), xem test_features_join_after_aggregate_is_one_to_one."""

from sales_forecast.ingestion.loaders import aggregate_to_store_date, join_features


def test_no_cartesian_assumption(sample_train):
    """Giả định A2: pipeline KHÔNG được giả định lưới đầy đủ Store x Dept.
    Số tổ hợp (Store, Dept) thực tế phải khớp đúng dữ liệu quan sát, không suy diễn thêm."""
    observed_pairs = sample_train[["Store", "Dept"]].drop_duplicates()
    assert len(observed_pairs) == 2  # (1,1) và (2,1) — không phải 4 = 2 Store x 2 Dept giả định


def test_features_join_row_count(sample_train, sample_features):
    """Giả định A3: join theo (Store, Date) không được nhân bản dòng của train."""
    joined = join_features(sample_train, sample_features)
    assert len(joined) == len(sample_train), \
        "Join features.csv theo (Store,Date) không được làm thay đổi số dòng của train"


def test_isholiday_consistency(sample_train, sample_features):
    """Giả định A4: IsHoliday ở train và features phải khớp nhau sau khi join,
    lệch nhau phải được phát hiện thay vì âm thầm lấy 1 nguồn.

    Chỉ so sánh các dòng có mặt ở cả 2 nguồn (features.csv không phủ hết mọi
    tuần trong fixture nhỏ) — dòng thiếu features là vấn đề coverage, không
    phải mismatch IsHoliday.
    """
    joined = join_features(sample_train, sample_features)
    both_present = joined.dropna(subset=["IsHoliday_features"])
    mismatches = both_present[both_present["IsHoliday_train"] != both_present["IsHoliday_features"]]
    assert len(mismatches) == 0, f"Phát hiện {len(mismatches)} dòng lệch IsHoliday giữa 2 nguồn"


def test_features_join_after_aggregate_is_one_to_one(sample_train, sample_test, sample_features):
    """A3 (mở rộng sau khi đổi đơn vị dự báo): sau aggregate_to_store_date,
    join giữa train (đã SUM theo Dept) và features.csv theo (Store, Date)
    không được nhân bản dòng — mỗi (Store, Date) chỉ join đúng 1 dòng features,
    quan hệ 1-1 thay vì N-1 như trước khi aggregate."""
    train_agg, _ = aggregate_to_store_date(sample_train, sample_test)
    joined = join_features(train_agg, sample_features)
    assert len(joined) == len(train_agg)
    # không có (Store,Date) nào bị nhân bản dòng sau join
    assert joined.duplicated(subset=["Store", "Date"]).sum() == 0
