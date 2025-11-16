# TiRex Cryptocurrency Fine-tuning

Complete toolkit for fine-tuning TiRex on your cryptocurrency data.

## 📁 Files in this Directory

- **`CRYPTO_FINETUNING_GUIDE.md`** - Complete step-by-step guide
- **`RETRAINING_GUIDE.md`** - Technical architecture details
- **`prepare_crypto_data.py`** - Data preparation script
- **`training_example.py`** - Fine-tuning training script
- **`README.md`** - This file

## 🚀 Quick Start (5 minutes)

### Step 1: Prepare Your Data

```bash
python prepare_crypto_data.py \
    --data-path ../../tests/data/btcusd_2022-06-01.joblib \
    --time-interval 60 \
    --output-dir ./prepared_data
```

**Expected output:**
```
✓ Loaded 8760 samples at 60-minute intervals
✓ Created 1095 sequences
✓ Data split: Train 80%, Val 10%, Test 10%
✓ Saved prepared data to ./prepared_data/crypto_prepared.joblib
```

### Step 2: Quick Test (2 minutes)

```bash
python training_example.py --quick-test
```

This runs 2 epochs on a small subset to verify everything works.

### Step 3: Full Fine-tuning (2-4 hours)

```bash
python training_example.py
```

### Step 4: Monitor Training

In a separate terminal:
```bash
tensorboard --logdir ./crypto_finetuned/logs
```

Open http://localhost:6006 in your browser.

## 📊 What Gets Created

After running the scripts, you'll have:

```
examples/training/
├── prepared_data/
│   └── crypto_prepared.joblib        # Preprocessed sequences
├── crypto_finetuned/
│   ├── checkpoints/
│   │   ├── crypto-tirex-TIMESTAMP-epoch=00-val_loss=0.0234.ckpt
│   │   ├── crypto-tirex-TIMESTAMP-epoch=01-val_loss=0.0198.ckpt
│   │   └── last.ckpt                 # Latest checkpoint
│   ├── logs/                         # TensorBoard logs
│   └── crypto_tirex_final_TIMESTAMP.ckpt  # Final model
```

## 🎯 Using Your Fine-tuned Model

### Option 1: In Your Existing Code

Update `opt_price_forecast.py`:

```python
from tirex import load_model

# Load fine-tuned model
model = load_model("./training/crypto_finetuned/checkpoints/best_model.ckpt")
```

### Option 2: Direct Usage

```python
from tirex import load_model
import torch

# Load model
model = load_model("./crypto_finetuned/checkpoints/best_model.ckpt")

# Make prediction
context = your_recent_prices[-512:]  # Last 512 points
context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

predictions = model.forecast(context=context_tensor, prediction_length=32)

# Extract quantiles
quantiles = predictions.cpu().numpy()[0]
median = quantiles[4, :]      # 50% quantile (median forecast)
lower = quantiles[0, :]       # 10% quantile (lower bound)
upper = quantiles[-1, :]      # 90% quantile (upper bound)
```

## 📈 Expected Performance

After fine-tuning on cryptocurrency data, you should see:

- **Training loss**: Decreases from ~0.05 to ~0.01
- **Validation MAE**: 0.01-0.03 (on normalized data)
- **Coverage**: ~80% (predictions within 80% interval)
- **Correlation**: 0.7-0.9 between predictions and actuals

## 🔧 Customization Options

### Different Time Intervals

```bash
# 15-minute intervals
python prepare_crypto_data.py --time-interval 15

# 5-minute intervals  
python prepare_crypto_data.py --time-interval 5
```

### Different Features

```bash
# Train on volume instead of price
python prepare_crypto_data.py --feature volume

# Train on high-low spread
python prepare_crypto_data.py --feature high_low_spread
```

### Adjust Training Parameters

```bash
# Longer training, larger batches
python training_example.py \
    --max-epochs 50 \
    --batch-size 64 \
    --learning-rate 2e-5

# Shorter context (faster, less memory)
python training_example.py \
    --context-length 256 \
    --prediction-length 16
```

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python training_example.py --batch-size 16

# Reduce context length
python training_example.py --context-length 256
```

### Training Not Improving

```bash
# Increase learning rate
python training_example.py --learning-rate 5e-5

# More epochs
python training_example.py --max-epochs 50
```

### No GPU Available

The code automatically falls back to CPU. Training will be slower but will work.

## 📚 Documentation

For detailed information, see:

- **`CRYPTO_FINETUNING_GUIDE.md`** - Complete tutorial with examples
- **`RETRAINING_GUIDE.md`** - Architecture and technical details

## 💡 Tips for Best Results

1. **Use clean data**: Remove outliers and handle missing values
2. **More data is better**: Combine multiple cryptocurrencies if available
3. **Monitor validation metrics**: Stop if overfitting occurs
4. **Try different features**: Volume, spreads, technical indicators
5. **Ensemble predictions**: Use multiple checkpoints and average

## 🎓 Next Steps

After successful fine-tuning:

1. Compare with pre-trained model performance
2. Integrate into your optimization pipeline (`opt_price_forecast.py`)
3. Experiment with different forecast horizons
4. Try multi-asset training (BTC, ETH, etc.)
5. Implement continual learning (retrain periodically)

## 📞 Need Help?

- Check `CRYPTO_FINETUNING_GUIDE.md` for detailed troubleshooting
- Review TiRex documentation at https://github.com/NX-AI/tirex
- File issues at the TiRex repository

## ✅ Checklist

Use this to track your progress:

- [ ] Installed dependencies (`conda activate tirex`, `pip install lightning`)
- [ ] Prepared data (`prepare_crypto_data.py`)
- [ ] Ran quick test (`training_example.py --quick-test`)
- [ ] Monitored with TensorBoard
- [ ] Completed full training
- [ ] Evaluated on test set
- [ ] Integrated into your workflow

Good luck! 🚀

