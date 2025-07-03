# WattWise Modal Deployment

This directory contains the files needed to deploy your WattWise Energy Consumption Forecasting app to [Modal](https://modal.com), a cloud platform for running Python code.

## 📋 Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install the Modal CLI tool
   ```bash
   pip install modal
   ```
3. **Authentication**: Set up your Modal credentials
   ```bash
   modal setup
   ```

## 📁 Files Overview

- **`modal_app.py`**: Main Modal application configuration with dependencies and file copying
- **`modal_streamlit.py`**: Streamlit app adapted for Modal deployment with custom styling
- **`README.md`**: This documentation file

## 🚀 Deployment

Deploy your app manually using the Modal CLI:
```bash
cd modal/
modal deploy modal_app.py
```

This will:
- ✅ Build the Modal environment with all dependencies
- ✅ Copy your data files and models to the cloud
- ✅ Deploy the Streamlit app with concurrent support (max 100 users)
- ✅ Provide the live app URL

## 📦 Dependencies

The Modal environment includes these packages:
- `streamlit` - Web app framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting library
- `plotly` - Interactive visualizations
- `seaborn` - Statistical data visualization
- `statsmodels` - Statistical modeling (SARIMAX)
- `scikit-learn` - Machine learning utilities
- `Pillow` - Image processing

## 📂 Required Files

The deployment copies these files to Modal:

### From your project:
- `src/utils.py` → `/root/src/utils.py`
- `data/data_cleaned.pkl` → `/root/data/data_cleaned.pkl`
- `data/data_split.pkl` → `/root/data/data_split.pkl`
- `data/feature_relevance_profile.pkl` → `/root/data/feature_relevance_profile.pkl`
- `models/sarimax_checkpoint.json` → `/root/models/sarimax_checkpoint.json`
- `image.jpg` → `/root/image.jpg`
- `.streamlit/config.toml` → `/root/.streamlit/config.toml`

### From modal directory:
- `modal_streamlit.py` → `/root/modal_streamlit.py`

## 🌐 Accessing Your App

After successful deployment, your app will be available at:
```
https://bbkhosseini--wattwise-energy-forecast-run.modal.run/
```

## ⚡ Performance Features

- **Concurrent Support**: The app supports up to 100 concurrent users
- **Custom Styling**: Enhanced sidebar width (400px) for better user experience
- **Cloud Scaling**: Modal automatically scales based on traffic

## 🔧 Customization

### Modify Dependencies
Edit the `pip_install()` section in `modal_app.py`:
```python
.pip_install(
    "streamlit>=1.46.0",
    "your-additional-package>=1.0.0",
    # Add more packages as needed
)
```

### Adjust Concurrency
Modify the concurrent users limit in `modal_app.py`:
```python
@app.function()
@modal.concurrent(max_inputs=100)  # Increase/decrease as needed
@modal.web_server(8000)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Modal CLI not found"**
   ```bash
   pip install modal
   ```

2. **"Modal authentication required"**
   ```bash
   modal setup
   ```

3. **"Missing required files"**
   - Make sure you're running from the correct directory
   - Check that all data files and models exist in the parent directory

4. **"Deployment failed"**
   - Check Modal logs: `modal logs wattwise-energy-forecast`
   - Verify file paths in `modal_app.py`

### Viewing Logs
```bash
modal logs wattwise-energy-forecast
```

### Stopping the App
```bash
modal app stop wattwise-energy-forecast
```

### Development Mode
For local development with hot reloading:
```bash
modal serve modal_app.py
```

## 💡 Tips

- **Development**: Use `modal serve` for local development with hot reloading
- **Monitoring**: Monitor app usage and costs in the Modal dashboard
- **Scaling**: The app supports up to 100 concurrent users by default
- **Costs**: Modal charges based on actual usage (CPU time, memory, etc.)
- **Styling**: Custom CSS is included for better sidebar appearance

## 📞 Support

- **Modal Documentation**: [docs.modal.com](https://docs.modal.com)
- **Modal Discord**: Join the Modal community for support
- **GitHub Issues**: Report issues in your project repository

---

🎉 **Happy Deploying!** Your WattWise app will be live on the cloud in minutes! 