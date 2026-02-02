# 🥁 Bonzo ⚖️ Jack the Dripper 🎨  
**A playful experiment about human randomness, rhythm vs chaos — powered by tiny neural networks.**

![Bonzo vs Jack – Promo](https://raw.githubusercontent.com/Dani-Luk/bonzo-vs-jack-the-dripper/refs/heads/main/assets/images/promo_bonzo_vs_jack_1200x800.png)

> *“You think you’re random. You’re not.”*

---

## 🎯 What is this?

**Bonzo vs Jack the Dripper** is a short, interactive **Red / Black (0–1)** game that explores how *predictable* human behavior really is.

There is:
- ❌ no strategy  
- ❌ no winning goal  
- ✅ just you, trying to be random  

Behind the scenes, tiny neural networks learn *your* patterns — not cards.

This project is about **human bias**, not gambling.

![Live App Screenshot](https://raw.githubusercontent.com/Dani-Luk/bonzo-vs-jack-the-dripper/refs/heads/main/assets/images/bonzo-vs-jack-the-dripper.queverything.com_screenshot.png)

---

## 🧠 The idea (rhythm vs chaos)

Why the names?

- **🥁 Bonzo** = John Bonham, Led Zeppelin's legendary drummer  → rhythm, repetition, steady patterns  

- **🎨 Jack the Dripper** = Jackson Pollock, famous for drip painting → chaos, irregularity, anti-patterns  

The AI measures things like:
- alternation vs repetition  
- streak lengths  
- short motifs (`001`, `010`, `111…`)  
- how quickly you drift into rhythm  

---
## 🎮 Two Versions
### 1️⃣ Python Proof of Concept 🧪
Before going to the browser, this project started as a **Python + Tkinter** PoC for: 
- validate the UX
- test online learning stability
- tune window sizes & training strategy
- debug “predict once, train once” logic

**Under the hood** two tiny models run **in parallel**:
| Model | What it’s good at |
|------|------------------|
| **GRU** | Learning short & mid-term habits |
| **Conv1D** | Detecting repeating micro-patterns |

Their predictions are combined into an **ensemble**, with a live slider to mix their influence.

The models:
- predict **once per click**
- train **online**, continuously
- adapt quickly with as little as **30–70 inputs**

### Features
- Append-only input (0 / 1, paste allowed)
- Conv1D + GRU models
- Ensemble mixing
- Real-time accuracy, SMA & cumulative plots
- Explicit TensorFlow memory management

### Run it locally
```bash
pip install tensorflow numpy matplotlib pillow
python PoC-RedBlack-GRU-Conv1D.py
```

📄 **See** ['PoC-RedBlack-GRU-Conv1D.py'](PoC-RedBlack-GRU-Conv1D.py) for full implementation.


---

### 2️⃣ Web Version (HTML/JavaScript)

A fully browser-based implementation using TensorFlow.js for maximum accessibility.

**Live Demo:** 👉 [**Try it here!**](https://bonzo-vs-jack-the-dripper.queverything.com/)

#### Features:
- 🌐 **Zero Installation**: Runs directly in any modern web browser
- 📱 **Mobile Friendly**: Responsive design with touch support
- 🎨 **Polished UI**: Smooth animations and professional styling
- 📊 **Real-time Charts**: Chart.js integration for live performance graphs
- 🎯 **AI Models**: Conv1D and GRU networks implemented in TensorFlow.js
- ⚖️ **Ensemble Mixing**: Interactive slider to blend predictions
- 🖼️ **Visual Theme**: Card-based interface with custom graphics

#### Technical Stack:
- **TensorFlow.js 4.15.0**: Neural network inference in the browser
- **Chart.js 4.4.1**: Real-time performance visualization

📄 **See** [`index.html'](index.html) for full JS implementation.

**Pro tip:** True randomness is harder than you think! Most people unconsciously alternate or create repeating patterns.  
Good luck! 🍀