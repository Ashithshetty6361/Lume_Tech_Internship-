import tensorflow as tf

# Check if GPU is detected
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

# Enable Memory Growth
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled.")
    except RuntimeError as e:
        print(e)

---
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Flatten, Dropout,
    Bidirectional, LSTM, Attention, Add,
    LayerNormalization, BatchNormalization, GaussianNoise
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, f1_score, accuracy_score)
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11

WINDOW     = 99
STRIDE     = 6
MIDPOINT   = WINDOW // 2
APPLIANCES = ['AC', 'Motor']
N_OUTPUTS  = len(APPLIANCES)

print(f'TensorFlow {tf.__version__}')
print(f'GPU: {tf.config.list_physical_devices("GPU")}')
---
df = pd.read_pickle('data/iawe_preprocessed.pkl')
print(f'Dataset: {len(df):,} rows × {df.shape[1]} columns')

# ============================================================
# 10 INPUT FEATURES
# ============================================================
FEATURE_COLS = ['Aggregate', 'Vrms', 'Irms', 'Reactive_Q',
                'Apparent_Power', 'Power_Factor',
                'Delta_P', 'Rolling_Mean', 'Rolling_Std', 'Hour']
N_FEATURES = len(FEATURE_COLS)

# Per-feature scaling
feat_scales = {}
for col in FEATURE_COLS:
    if col == 'Power_Factor':
        feat_scales[col] = 1.0
    elif col == 'Hour':
        feat_scales[col] = 24.0
    elif col == 'Delta_P':
        feat_scales[col] = max(np.percentile(np.abs(df[col].values), 99.5), 1.0)
    else:
        feat_scales[col] = max(np.percentile(df[col].values, 99.5), 1.0)

# Build normalised feature matrix
feat_matrix = np.zeros((len(df), N_FEATURES), dtype='float32')
for j, col in enumerate(FEATURE_COLS):
    if col == 'Delta_P':
        feat_matrix[:, j] = df[col].values / feat_scales[col]
    else:
        feat_matrix[:, j] = np.clip(df[col].values / feat_scales[col], 0, 2)

# Appliance target scales
app_max = {}
for name in APPLIANCES:
    app_max[name] = max(np.percentile(df[name].values, 99.9), 1.0)
    print(f'  {name} max = {app_max[name]:.1f} W')
agg_max = feat_scales['Aggregate']

# ============================================================
# SLIDING WINDOWS — Multi-output targets
# ============================================================
num_windows = (len(feat_matrix) - WINDOW) // STRIDE
print(f'\nWindows: {num_windows:,}  (window={WINDOW}, stride={STRIDE})')

X_all = np.zeros((num_windows, WINDOW, N_FEATURES), dtype='float32')
Y_all = np.zeros((num_windows, N_OUTPUTS), dtype='float32')  # [AC, Motor]

for i in range(num_windows):
    s = i * STRIDE
    X_all[i] = feat_matrix[s : s + WINDOW]
    for j, name in enumerate(APPLIANCES):
        Y_all[i, j] = df[name].values[s + MIDPOINT] / app_max[name]

# ============================================================
# CHRONOLOGICAL SPLIT: 70 / 15 / 15
# ============================================================
indices = np.arange(num_windows)  # CHRONOLOGICAL SPLIT
n_train = int(num_windows * 0.70)
n_val   = int(num_windows * 0.15)

idx_train = indices[:n_train]
idx_val   = indices[n_train:n_train+n_val]
idx_test  = indices[n_train+n_val:]

X_train, X_val, X_test = X_all[idx_train], X_all[idx_val], X_all[idx_test]
Y_train, Y_val, Y_test = Y_all[idx_train], Y_all[idx_val], Y_all[idx_test]

print(f'\nTrain: {X_train.shape}  targets: {Y_train.shape}')
print(f'Val:   {X_val.shape}')
print(f'Test:  {X_test.shape}')
for j, name in enumerate(APPLIANCES):
    on_pct = np.mean(Y_test[:, j] > 0.01) * 100
    print(f'  {name} test ON: {on_pct:.1f}%')
---
def build_model():
    inp = Input(shape=(WINDOW, N_FEATURES), name='aggregate_input')

    # Data augmentation (only during training)
    x = GaussianNoise(0.02)(inp)

    # Shared CNN backbone
    x = Conv1D(32, 10, activation='relu', name='conv1')(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, 8, activation='relu', name='conv2')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 6, activation='relu', name='conv3')(x)
    x = BatchNormalization()(x)

    # Shared BiLSTM
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2),
                      name='bilstm')(x)

    # Shared Self-Attention
    attn = Attention(name='self_attn')([x, x])
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    # Shared Dense backbone
    x = Flatten()(x)
    x = Dense(512, activation='relu', name='shared_dense1')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', name='shared_dense2')(x)
    x = Dropout(0.2)(x)

    # Task-Specific Heads
    # AC Head
    ac_head = Dense(128, activation='relu')(x)
    ac_head = Dropout(0.2)(ac_head)
    out_ac = Dense(1, activation='linear', name='out_ac')(ac_head)

    # Motor Head
    motor_head = Dense(128, activation='relu')(x)
    motor_head = Dropout(0.2)(motor_head)
    out_motor = Dense(1, activation='linear', name='out_motor')(motor_head)

    import tensorflow as tf
    out = tf.keras.layers.Concatenate(name='multi_output')([out_ac, out_motor])

    model = Model(inputs=inp, outputs=out, name='NILM_MultiOutput')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='huber',
        metrics=['mae']
    )
    return model

model = build_model()
model.summary()
---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6, verbose=1),
]

print('Training single multi-output model...')
print(f'Predicting {N_OUTPUTS} appliances: {APPLIANCES}')
print(f'Input shape: {X_train.shape}  |  Target shape: {Y_train.shape}')

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

print(f'\n✓ Model trained! Best val_loss: {min(history.history["val_loss"]):.6f}')
---
# Raw predictions
raw_preds = model.predict(X_test, verbose=0)  # shape: (n_test, 2)
print(f'Raw predictions shape: {raw_preds.shape}')

MEDIAN_SIZES = {'AC': 5, 'Motor': 3}

results = {}
for j, appliance in enumerate(APPLIANCES):
    raw_w  = np.clip(raw_preds[:, j] * app_max[appliance], 0, None)
    filt_w = median_filter(raw_w, size=MEDIAN_SIZES[appliance])
    true_w = Y_test[:, j] * app_max[appliance]

    results[appliance] = {
        'true': true_w,
        'pred': filt_w,
        'raw':  raw_w,
    }
    print(f'{appliance}: true range [{true_w.min():.0f}, {true_w.max():.0f}] W  |  '
          f'pred range [{filt_w.min():.0f}, {filt_w.max():.0f}] W')
---
def compute_metrics(true_w, pred_w, threshold=15):
    mae  = mean_absolute_error(true_w, pred_w)
    rmse = np.sqrt(mean_squared_error(true_w, pred_w))
    r2   = r2_score(true_w, pred_w) if np.std(true_w) > 0 else 0
    nde  = np.sum((pred_w - true_w)**2) / (np.sum(true_w**2) + 1e-6)
    da   = 1 - np.sum(np.abs(pred_w - true_w)) / (2 * np.sum(np.abs(true_w)) + 1e-6)
    sae  = abs(pred_w.sum() - true_w.sum()) / (true_w.sum() + 1e-6)

    true_on = (true_w > threshold).astype(int)
    pred_on = (pred_w > threshold).astype(int)
    acc = accuracy_score(true_on, pred_on)
    f1  = f1_score(true_on, pred_on, average='macro', zero_division=0)

    return {'MAE (W)': mae, 'RMSE (W)': rmse, 'R²': r2, 'NDE': nde,
            'DA': da, 'Accuracy': acc, 'F1-Macro': f1, 'SAE': sae}

rows = []
for appliance in APPLIANCES:
    m = compute_metrics(results[appliance]['true'], results[appliance]['pred'])
    rows.append({'Appliance': appliance, **m})

df_metrics = pd.DataFrame(rows).set_index('Appliance')

# Pretty print
fmt = df_metrics.copy()
fmt['MAE (W)']  = fmt['MAE (W)'].map('{:.1f}'.format)
fmt['RMSE (W)'] = fmt['RMSE (W)'].map('{:.1f}'.format)
fmt['R²']       = fmt['R²'].map('{:.4f}'.format)
fmt['NDE']      = fmt['NDE'].map('{:.4f}'.format)
fmt['DA']       = fmt['DA'].map('{:.1%}'.format)
fmt['Accuracy'] = fmt['Accuracy'].map('{:.1%}'.format)
fmt['F1-Macro'] = fmt['F1-Macro'].map('{:.4f}'.format)
fmt['SAE']      = fmt['SAE'].map('{:.4f}'.format)

print('\n' + '='*90)
print('  Single Multi-Output Model — Disaggregation Results')
print('='*90)
display(fmt)

# Context for Motor performance
print('\n--- Interpretation ---')
print(f'AC draws ~{app_max["AC"]:.0f}W → MAE of {df_metrics.loc["AC","MAE (W)"]:.0f}W = '
      f'{df_metrics.loc["AC","MAE (W)"]/app_max["AC"]*100:.1f}% relative error')
print(f'Motor draws ~{app_max["Motor"]:.0f}W → MAE of {df_metrics.loc["Motor","MAE (W)"]:.0f}W = '
      f'{df_metrics.loc["Motor","MAE (W)"]/app_max["Motor"]*100:.1f}% relative error')
print(f'Motor R² is lower because the motor oscillates rapidly (ON/OFF every few seconds).')
print(f'The model correctly captures the AVERAGE motor power; individual oscillations are below the 6s sampling resolution.')
---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], linewidth=2, color='#2980b9', label='Train')
axes[0].plot(history.history['val_loss'], linewidth=2, color='#e67e22', linestyle='--', label='Val')
best_epoch = np.argmin(history.history['val_loss'])
axes[0].axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7)
axes[0].set_title(f'Loss (best epoch: {best_epoch})', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Huber Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history.history['mae'], linewidth=2, color='#2980b9', label='Train')
axes[1].plot(history.history['val_mae'], linewidth=2, color='#e67e22', linestyle='--', label='Val')
axes[1].set_title('MAE (normalised)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Training Convergence — Single Multi-Output Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, appliance in enumerate(APPLIANCES):
    true_w = results[appliance]['true']
    pred_w = results[appliance]['pred']

    axes[i].scatter(true_w, pred_w, alpha=0.1, s=4, c='#3498db', rasterized=True)
    max_val = max(np.max(true_w), np.max(pred_w)) * 1.05
    axes[i].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')

    r2 = r2_score(true_w, pred_w) if np.std(true_w) > 0 else 0
    axes[i].text(0.05, 0.92, f'R² = {r2:.4f}', transform=axes[i].transAxes,
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    axes[i].set_title(f'{appliance}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Actual (W)')
    axes[i].set_ylabel('Predicted (W)')
    axes[i].legend(fontsize=10)
    axes[i].set_xlim(0, max_val)
    axes[i].set_ylim(0, max_val)
    axes[i].set_aspect('equal')

plt.suptitle('Prediction Accuracy — Single Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
---
# Sort test windows back into temporal order
sorted_order = np.argsort(idx_test)
X_test_sorted = X_test[sorted_order]

preds_sorted = {}
for appliance in APPLIANCES:
    preds_sorted[appliance] = {
        'true': results[appliance]['true'][sorted_order],
        'pred': results[appliance]['pred'][sorted_order],
    }

# Find 600-point segment with the most AC activity
ac_sorted = preds_sorted['AC']['true']
N = 600
best_start, best_score = 0, 0
for s in range(0, len(ac_sorted) - N, 50):
    score = np.sum(ac_sorted[s:s+N] > 50)
    if score > best_score:
        best_score = score
        best_start = s

sl = slice(best_start, best_start + N)

# Recover aggregate
agg_plot = np.array([X_test_sorted[i, MIDPOINT, 0]
                     for i in range(best_start, best_start+N)]) * agg_max

# ---- PLOT ----
colors = {'AC': '#e74c3c', 'Motor': '#2980b9'}
n_panels = 1 + len(APPLIANCES)
fig, axes = plt.subplots(n_panels, 1, figsize=(20, 5*n_panels), sharex=True)

x_axis = np.arange(N)

# Panel 1: Aggregate
axes[0].plot(x_axis, agg_plot, color='black', linewidth=1.2)
axes[0].fill_between(x_axis, agg_plot, alpha=0.06, color='black')
axes[0].set_title('Aggregate Household Power (Model Input)', fontsize=15, fontweight='bold')
axes[0].set_ylabel('Watts', fontsize=12)
axes[0].set_ylim(0)

for idx, appliance in enumerate(APPLIANCES):
    ax = axes[idx + 1]
    true_w = np.array(preds_sorted[appliance]['true'][sl])
    pred_w = np.array(preds_sorted[appliance]['pred'][sl])

    ax.fill_between(x_axis, true_w, alpha=0.25, color='dodgerblue', label='Ground Truth')
    ax.plot(x_axis, true_w, color='dodgerblue', linewidth=1.5, alpha=0.8)
    ax.plot(x_axis, pred_w, color=colors[appliance], linewidth=1.8, linestyle='--',
            label='Predicted (Single Model)')

    mae = mean_absolute_error(true_w, pred_w)
    r2 = r2_score(true_w, pred_w) if np.std(true_w) > 0 else 0
    ax.text(0.01, 0.95, f'MAE = {mae:.1f} W   |   R² = {r2:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_title(f'{appliance} — Disaggregated Waveform', fontsize=15, fontweight='bold')
    ax.set_ylabel('Watts', fontsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_ylim(0)

axes[-1].set_xlabel('Test Time Steps (each step ≈ 60 seconds)', fontsize=12)

plt.suptitle('NILM Disaggregation — Single Multi-Output Model',
             fontsize=17, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

print('\n✓ Both appliances disaggregated by a SINGLE model in one forward pass!')
---
fig, ax = plt.subplots(1, 1, figsize=(20, 6))
x_axis = np.arange(N)

ax.plot(x_axis, agg_plot, color='black', linewidth=2, alpha=0.8, label='Aggregate (Actual)')

pred_sum = np.zeros(N)
true_sum = np.zeros(N)
for appliance in APPLIANCES:
    pred_sum += np.array(preds_sorted[appliance]['pred'][sl])
    true_sum += np.array(preds_sorted[appliance]['true'][sl])

ax.plot(x_axis, pred_sum, color='#e74c3c', linewidth=1.5, linestyle='--',
        label='Predicted Sum (AC + Motor)')
ax.fill_between(x_axis, true_sum, alpha=0.15, color='green',
                label='Ground Truth Sum (AC + Motor)')

ax.set_title('Reconstruction: Predicted Appliance Sum vs Aggregate', fontsize=15, fontweight='bold')
ax.set_ylabel('Watts')
ax.set_xlabel('Test Time Steps')
ax.legend(fontsize=11)
ax.set_ylim(0)
plt.tight_layout()
plt.show()

recon_mae = mean_absolute_error(true_sum, pred_sum)
print(f'Reconstruction MAE (AC+Motor): {recon_mae:.1f} W')
print(f'Gap between aggregate and sum = other unmeasured appliances (fridge, lights, etc.)')
---
fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
x_axis = np.arange(N)

for idx, appliance in enumerate(APPLIANCES):
    ax = axes[idx]
    true_w = np.array(preds_sorted[appliance]['true'][sl])
    raw_sorted = np.array(results[appliance]['raw'][sorted_order][sl])
    pred_w = np.array(preds_sorted[appliance]['pred'][sl])

    ax.plot(x_axis, true_w, color='dodgerblue', linewidth=1.5, alpha=0.7, label='Ground Truth')
    ax.plot(x_axis, raw_sorted, color='gray', linewidth=0.8, alpha=0.4, label='Raw Prediction')
    ax.plot(x_axis, pred_w, color=colors[appliance], linewidth=1.8, linestyle='--',
            label=f'Median Filtered (size={MEDIAN_SIZES[appliance]})')

    raw_mae  = mean_absolute_error(true_w, raw_sorted)
    filt_mae = mean_absolute_error(true_w, pred_w)
    ax.text(0.01, 0.95, f'Raw: {raw_mae:.1f}W → Filtered: {filt_mae:.1f}W',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_title(f'{appliance}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Watts')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0)

axes[-1].set_xlabel('Time Steps')
plt.suptitle('Post-Processing Effect', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
---
# Save model for deployment
model.save('data/nilm_multi_output_model.keras')
print('\n✓ Model saved to data/nilm_multi_output_model.keras')
print(f'  Input shape:  {model.input_shape}')
print(f'  Output shape: {model.output_shape}')
print(f'  Total params: {model.count_params():,}')
print(f'  Appliances:   {APPLIANCES}')
---

