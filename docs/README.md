# Wunder Challenge: LOB Predictorium - Documentation

Complete documentation for the Wunder Neural Network Predictorium competition.

## 🚀 Getting Started

Start here if you're new to the competition:

- **[Quick Start Guide](QUICK_START.md)** - Get up and running in minutes with the starter pack

## 📖 Core Documentation

### Understanding the Competition

- **[Data Overview](DATA.md)** - Detailed breakdown of features, sequences, and evaluation metrics
- **[Competition Rules](RULES.md)** - Fair play guidelines and requirements
- **[Timeline](TIMELINE.md)** - Key dates and deadlines
- **[Prizes](PRIZES.md)** - $13,600 prize pool and verification process

### Technical Guides

- **[Submission Guide](SUBMISSION_GUIDE.md)** - How to package and submit your solution
  - Code requirements
  - Resource limits
  - Docker environment details
  - Packaging instructions

## 💬 Help & Support

- **[FAQ](FAQ.md)** - Frequently asked questions
- **[Get Help](GET_HELP.md)** - Discord and email support

## 🔗 External Resources

- [Competition Website](https://predictorium.wundernn.io/)
- [Leaderboard](https://predictorium.wundernn.io/leaderboard)
- [Submit](https://predictorium.wundernn.io/submit)
- [Discord Community](https://predictorium.wundernn.io/discord)

---

## Competition Overview

### The Challenge

Predict future market indicators (t0, t1) from sequences of Limit Order Book (LOB) states and recent trade data. This is a sequence modeling task from the domain of high-frequency trading.

### Key Facts

- **Dataset**: 10,721 training sequences, 1,444 validation sequences
- **Sequence Length**: 1,000 steps (99 warm-up + 901 scored predictions)
- **Evaluation**: Weighted Pearson Correlation Coefficient
- **Submission**: Code competition (CPU-only, 90-minute time limit)
- **Prize Pool**: $13,600 USDT for top 8 participants

### Quick Links

- 📦 [Get Starter Pack](QUICK_START.md#1-get-the-starter-pack)
- 🏃 [Run Example Solution](QUICK_START.md#4-run-the-example-solution)
- 📤 [Submit Your Solution](SUBMISSION_GUIDE.md#submit-your-package)

Good luck! 🚀

