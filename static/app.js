// AI Demand Forecasting Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    const predictionResult = document.getElementById('predictionResult');
    const predictedValue = document.getElementById('predictedValue');

    // Handle form submission
    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        const submitBtn = predictionForm.querySelector('.btn-primary');
        const originalText = submitBtn.textContent;
        submitBtn.innerHTML = '<span class="loading"></span> Predicting...';
        submitBtn.disabled = true;
        
        // Hide previous results
        predictionResult.style.display = 'none';
        
        try {
            // Collect form data
            const formData = new FormData(predictionForm);
            const predictionData = {
                item_id: formData.get('item_id'),
                store_id: formData.get('store_id'),
                dept_id: formData.get('dept_id'),
                sell_price: parseFloat(formData.get('sell_price')),
                prediction_date: formData.get('prediction_date'),
                has_event: formData.get('has_event') ? 1 : 0
            };
            
            // Send prediction request
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(predictionData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                // Display prediction result
                predictedValue.textContent = Math.round(result.prediction * 100) / 100;
                predictionResult.style.display = 'block';
                
                // Smooth scroll to result
                predictionResult.scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'center'
                });
                
                // Add success animation
                predictionResult.style.animation = 'none';
                setTimeout(() => {
                    predictionResult.style.animation = 'slideInUp 0.5s ease-out';
                }, 10);
                
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
            
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Prediction failed: ' + error.message);
        } finally {
            // Restore button state
            submitBtn.textContent = originalText;
            submitBtn.disabled = false;
        }
    });
    
    // Add date validation (prevent past dates)
    const dateInput = document.getElementById('prediction_date');
    if (dateInput) {
        const today = new Date().toISOString().split('T')[0];
        dateInput.setAttribute('min', today);
        
        // Set default to tomorrow
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        dateInput.value = tomorrow.toISOString().split('T')[0];
    }
    
    // Auto-generate category from department
    const deptSelect = document.getElementById('dept_id');
    if (deptSelect) {
        deptSelect.addEventListener('change', function() {
            const deptValue = this.value;
            if (deptValue.includes('HOBBIES')) {
                // Auto-set category for hobbies
            } else if (deptValue.includes('HOUSEHOLD')) {
                // Auto-set category for household
            } else if (deptValue.includes('FOODS')) {
                // Auto-set category for foods
            }
        });
    }
    
    // Price input validation
    const priceInput = document.getElementById('sell_price');
    if (priceInput) {
        priceInput.addEventListener('input', function() {
            const value = parseFloat(this.value);
            if (value < 0) {
                this.value = 0;
            } else if (value > 1000) {
                this.value = 1000;
            }
        });
    }
    
    // Add hover effects to metric cards
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05) rotate(1deg)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1) rotate(0deg)';
        });
    });
    
    // Add loading animation for chart
    const chartCanvas = document.getElementById('importanceChart');
    if (chartCanvas) {
        // Add a subtle animation when chart loads
        chartCanvas.style.opacity = '0';
        setTimeout(() => {
            chartCanvas.style.transition = 'opacity 1s ease-in-out';
            chartCanvas.style.opacity = '1';
        }, 500);
    }
    
    // Smooth reveal animation for cards
    const cards = document.querySelectorAll('.card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(card);
    });
});

// Add custom animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    .loading {
        animation: spin 1s linear infinite, pulse 2s ease-in-out infinite;
    }
`;
document.head.appendChild(style);