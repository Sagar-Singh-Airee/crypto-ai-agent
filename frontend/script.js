// Enhanced Navbar scroll effect
const navbar = document.getElementById('navbar');
let lastScrollY = window.scrollY;

window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 50);
  
  // Hide navbar on scroll down, show on scroll up
  if (window.scrollY > lastScrollY && window.scrollY > 100) {
    navbar.style.transform = 'translateY(-100%)';
  } else {
    navbar.style.transform = 'translateY(0)';
  }
  lastScrollY = window.scrollY;
});

// Enhanced Mobile menu toggle
const mobileToggle = document.querySelector('.mobile-toggle');
const navLinks = document.querySelector('.nav-links');
mobileToggle.addEventListener('click', () => {
  mobileToggle.classList.toggle('active');
  navLinks.classList.toggle('active');
  document.body.style.overflow = navLinks.classList.contains('active') ? 'hidden' : '';
});

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-links a').forEach(link => {
  link.addEventListener('click', () => {
    mobileToggle.classList.remove('active');
    navLinks.classList.remove('active');
    document.body.style.overflow = '';
  });
});

// Enhanced Scroll reveal animation
const revealElements = document.querySelectorAll('.scroll-reveal');
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('active');
      // Add staggered delay for multiple elements
      if (entry.target.parentElement.classList.contains('steps-container') || 
          entry.target.parentElement.classList.contains('news-items')) {
        const index = Array.from(entry.target.parentElement.children).indexOf(entry.target);
        entry.target.style.transitionDelay = `${index * 0.1}s`;
      }
    }
  });
}, { 
  threshold: 0.1,
  rootMargin: '0px 0px -50px 0px'
});

revealElements.forEach(el => revealObserver.observe(el));

// Enhanced Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(anchor.getAttribute('href'));
    if (target) {
      const offsetTop = target.getBoundingClientRect().top + window.pageYOffset - 80;
      window.scrollTo({
        top: offsetTop,
        behavior: 'smooth'
      });
    }
  });
});

// Enhanced Demo analysis functionality with loading state
const analyzeBtn = document.getElementById('analyze-btn');
const newsInput = document.getElementById('news-input');

analyzeBtn.addEventListener('click', async () => {
  const originalText = analyzeBtn.innerHTML;
  const inputText = newsInput.value.trim();
  
  if (!inputText) {
    showNotification('Please enter a news headline to analyze', 'error');
    return;
  }
  
  // Show loading state
  analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
  analyzeBtn.disabled = true;
  
  // Simulate API call delay
  await new Promise(resolve => setTimeout(resolve, 1500));
  
  // Enhanced sentiment analysis with more factors
  const analysisResult = analyzeSentiment(inputText);
  
  // Update the UI with the result
  updateAnalysisResult(analysisResult);
  
  // Reset button
  analyzeBtn.innerHTML = originalText;
  analyzeBtn.disabled = false;
  
  // Show success notification
  showNotification('Analysis complete!', 'success');
});

// Enhanced sentiment analysis function
function analyzeSentiment(text) {
  const lowerText = text.toLowerCase();
  let sentiment = 'positive';
  let confidence = 87;
  let prediction = '+2.8%';
  let impact = 'High';
  let timeframe = '24-48 hours';
  let explanation = 'Based on historical patterns, this type of news typically leads to short-term price increases.';
  
  // More sophisticated keyword analysis
  const positiveKeywords = ['adoption', 'partnership', 'integration', 'approval', 'surge', 'growth', 'bullish', 'rally'];
  const negativeKeywords = ['regulation', 'ban', 'crackdown', 'hack', 'security breach', 'exploit', 'crash', 'bearish'];
  const neutralKeywords = ['update', 'maintenance', 'partnership', 'announcement'];
  
  let positiveCount = positiveKeywords.filter(word => lowerText.includes(word)).length;
  let negativeCount = negativeKeywords.filter(word => lowerText.includes(word)).length;
  let neutralCount = neutralKeywords.filter(word => lowerText.includes(word)).length;
  
  // Determine sentiment based on keyword counts
  if (negativeCount > positiveCount) {
    sentiment = 'negative';
    confidence = 75 + Math.min(negativeCount * 5, 15);
    prediction = '-1.5%';
    impact = 'Medium';
    explanation = 'Historical data shows this type of news often causes temporary price declines.';
  } else if (positiveCount > negativeCount) {
    sentiment = 'positive';
    confidence = 80 + Math.min(positiveCount * 4, 15);
    prediction = '+2.1%';
    impact = 'High';
    explanation = 'Positive developments typically lead to moderate price increases within 24-48 hours.';
  } else {
    sentiment = 'neutral';
    confidence = 60;
    prediction = '+0.5%';
    impact = 'Low';
    explanation = 'Mixed or neutral news typically has minimal short-term impact on prices.';
  }
  
  // Special cases
  if (lowerText.includes('bitcoin etf') || lowerText.includes('etf approval')) {
    sentiment = 'positive';
    confidence = 94;
    prediction = '+3.5%';
    impact = 'Very High';
    explanation = 'ETF approvals have historically led to significant positive price movements in the following days.';
  }
  
  if (lowerText.includes('regulation') && lowerText.includes('positive')) {
    sentiment = 'positive';
    confidence = 82;
    prediction = '+1.8%';
    impact = 'Medium';
    explanation = 'Positive regulatory clarity typically boosts market confidence and prices.';
  }
  
  return {
    sentiment,
    confidence,
    prediction,
    impact,
    timeframe,
    explanation
  };
}

// Enhanced UI update function
function updateAnalysisResult(result) {
  const predictionValue = document.querySelector('.prediction-value');
  const confidenceScore = document.querySelector('.confidence-score');
  const breakdownValues = document.querySelectorAll('.breakdown-value');
  const analysisExplanation = document.querySelector('.analysis-explanation p');
  
  // Update prediction value
  predictionValue.textContent = result.prediction;
  predictionValue.className = `prediction-value ${result.sentiment}`;
  
  // Update confidence
  confidenceScore.textContent = `${result.confidence}% Confidence`;
  confidenceScore.style.background = result.sentiment === 'positive' 
    ? 'var(--accent-green-light)' 
    : result.sentiment === 'negative' 
    ? 'var(--accent-red-light)' 
    : 'var(--accent-blue-light)';
  confidenceScore.style.color = result.sentiment === 'positive' 
    ? 'var(--accent-green)' 
    : result.sentiment === 'negative' 
    ? 'var(--accent-red)' 
    : 'var(--accent-blue)';
  
  // Update breakdown
  breakdownValues[0].textContent = result.sentiment === 'positive' ? '+0.84' : result.sentiment === 'negative' ? '-0.76' : '+0.12';
  breakdownValues[0].className = `breakdown-value ${result.sentiment}`;
  breakdownValues[1].textContent = result.impact;
  breakdownValues[2].textContent = result.timeframe;
  
  // Update explanation
  analysisExplanation.innerHTML = `<strong>Analysis:</strong> ${result.explanation}`;
}

// Notification system
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `notification notification-${type}`;
  notification.innerHTML = `
    <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
    <span>${message}</span>
    <button onclick="this.parentElement.remove()">
      <i class="fas fa-times"></i>
    </button>
  `;
  
  // Add styles
  notification.style.cssText = `
    position: fixed;
    top: 100px;
    right: 20px;
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-color);
    border-left: 4px solid ${type === 'success' ? 'var(--accent-green)' : type === 'error' ? 'var(--accent-red)' : 'var(--accent-blue)'};
    padding: 1rem 1.5rem;
    border-radius: 12px;
    box-shadow: var(--shadow-glow);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    z-index: 10000;
    animation: slideInRight 0.3s ease-out;
    max-width: 400px;
  `;
  
  document.body.appendChild(notification);
  
  // Auto remove after 5 seconds
  setTimeout(() => {
    if (notification.parentElement) {
      notification.style.animation = 'slideOutRight 0.3s ease-in';
      setTimeout(() => notification.remove(), 300);
    }
  }, 5000);
}

// Add CSS for notifications
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
  @keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
  
  @keyframes slideOutRight {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(100%); opacity: 0; }
  }
  
  .notification button {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 0.25rem;
    border-radius: 4px;
    transition: all 0.3s ease;
  }
  
  .notification button:hover {
    background: rgba(255,255,255,0.1);
    color: var(--text-primary);
  }
`;
document.head.appendChild(notificationStyles);

// Enhanced live news updates with animation
function updateNewsFeed() {
  const newsItems = document.querySelector('.news-items');
  const lastUpdated = document.querySelector('.last-updated');
  
  // In a real implementation, this would fetch from your API
  const now = new Date();
  lastUpdated.innerHTML = `<i class="fas fa-sync-alt"></i> Updated ${now.getMinutes()} min ago`;
  
  // Add subtle animation to news items
  document.querySelectorAll('.news-item').forEach((item, index) => {
    setTimeout(() => {
      item.style.transform = 'translateX(0)';
      item.style.opacity = '1';
    }, index * 100);
  });
}

// Initialize news items with staggered animation
document.querySelectorAll('.news-item').forEach((item, index) => {
  item.style.transform = 'translateX(20px)';
  item.style.opacity = '0';
  item.style.transition = `all 0.5s ease ${index * 0.1}s`;
});

// Update news feed every 2 minutes
setInterval(updateNewsFeed, 120000);

// Initialize on load
window.addEventListener('load', () => {
  // Animate news items in
  setTimeout(() => {
    document.querySelectorAll('.news-item').forEach((item, index) => {
      setTimeout(() => {
        item.style.transform = 'translateX(0)';
        item.style.opacity = '1';
      }, index * 100);
    });
  }, 1000);
  
  // Add hover effects to interactive elements
  document.querySelectorAll('.step-card, .metric-card, .news-item').forEach(card => {
    card.addEventListener('mouseenter', () => {
      card.style.transform = 'translateY(-5px)';
    });
    
    card.addEventListener('mouseleave', () => {
      card.style.transform = 'translateY(0)';
    });
  });
});

// Enhanced input validation
newsInput.addEventListener('input', (e) => {
  const value = e.target.value;
  if (value.length > 200) {
    e.target.value = value.slice(0, 200);
    showNotification('Headline truncated to 200 characters', 'info');
  }
});

// Add keyboard shortcut for analysis (Ctrl/Cmd + Enter)
newsInput.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    analyzeBtn.click();
  }
});

// Enhanced sentiment chart interaction
document.querySelectorAll('.sentiment-bar').forEach(bar => {
  bar.addEventListener('click', () => {
    const coin = bar.querySelector('.bar-label').textContent;
    const prediction = bar.querySelector('.bar-value').textContent;
    showNotification(`Selected ${coin}: Predicted ${prediction} movement`, 'info');
  });
});

// Add real-time market data simulation
function simulateMarketData() {
  const metrics = document.querySelectorAll('.market-metric .metric-value');
  if (metrics.length > 0) {
    // Simulate small price changes
    const change = (Math.random() - 0.5) * 0.5;
    const newValue = 1.72 + change;
    metrics[1].innerHTML = `$${newValue.toFixed(2)}T <span class="${change >= 0 ? 'positive' : 'negative'}">${change >= 0 ? '+' : ''}${change.toFixed(1)}%</span>`;
  }
}

// Update market data every 30 seconds
setInterval(simulateMarketData, 30000);