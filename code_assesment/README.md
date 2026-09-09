"""
Hotel Booking Cancellation Prediction: A/B Test of a proposed solution vs. a published
Kaggle-style baseline
========================================================================================
Dataset: "Hotel Booking Demand" (Antonio, Almeida & Nunes, 2019; published on Kaggle by
Jesse Mostipak). 119,390 bookings across a Resort Hotel and a City Hotel (Portugal,
Jul 2015 - Aug 2017). Target: is_canceled (1 = booking was cancelled).

Business relevance: booking-cancellation prediction is a core revenue-management problem
for a hotel operator (Hilton and peers use models like this to drive overbooking limits,
targeted deposit/cancellation-policy offers, and proactive outreach to at-risk
reservations).

Leakage note (applies to BOTH models): `reservation_status` and `reservation_status_date`
directly encode the outcome (reservation_status == "Canceled" iff is_canceled == 1) and are
recorded *after* the booking is resolved, so they are dropped from every feature set below.
This is a well-known trap in public notebooks on this dataset; some published solutions do
NOT drop these and report unrealistically high (>95%) accuracy as a result.

BASELINE (replicates the standard published approach seen across most public
Kaggle/GitHub notebooks for this dataset, e.g. "Optimizing Hotel Revenue Through Booking
Cancellation Prediction" and similar RandomForest-based writeups):
  - Feature set: hotel, lead_time, arrival_date_year/month/week/day, stays_in_weekend_nights,
    stays_in_week_nights, adults, children, babies, meal, country, market_segment,
    distribution_channel, is_repeated_guest, previous_cancellations, deposit_type
  - country: one-hot encoded as-is (177 raw categories -> ~177 sparse columns), the most
    common treatment in public notebooks
  - Missing children -> 0; missing country -> mode
  - Model: RandomForestClassifier, default-ish hyperparameters

PROPOSED solution:
  - Same leakage-safe base, plus domain-driven engineered features a revenue-management
    team would actually want:
      * total_nights, total_guests, is_family
      * prior_cancel_rate = previous_cancellations / (previous_cancellations +
        previous_bookings_not_canceled + 1) -- a loyalty/risk signal the baseline
        feature set doesn't construct
      * room_type_mismatch = assigned_room_type != reserved_room_type (a known signal:
        a reassigned room usually means the guest already checked in / hotel accommodated
        them, correlated with lower cancellation)
      * booking_changes, total_of_special_requests, required_car_parking_spaces,
        days_in_waiting_list, adr (log-transformed)
      * agent: missing -> "None" category, kept as frequency-encoded (not one-hot, since
        it has 333 raw levels)
  - High-cardinality categoricals (country, agent) frequency-encoded instead of one-hot,
    avoiding the ~500-column sparse blow-up the baseline creates
  - Model: XGBoost gradient boosting (histogram tree method, depth 5, 200 trees)

A/B test:
  H0: proposed solution's mean CV accuracy == baseline's mean CV accuracy
  H1 (one-sided): proposed > baseline
  Same 5-fold Stratified CV splits (paired) for both models; metrics: Accuracy, Precision,
  Recall, F1, ROC-AUC. Significance via paired t-test + bootstrap 95% CI.
  (5 folds, not 10, because n=119,390 already gives each fold ~24k rows -- ample for
  stable per-fold estimates -- and it keeps runtime reasonable for gradient boosting.)
"""