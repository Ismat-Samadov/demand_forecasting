// AI Demand Forecasting Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    const predictionResult = document.getElementById('predictionResult');
    const predictedValue = document.getElementById('predictedValue');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const businessImpact = document.getElementById('businessImpact');
    const recommendation = document.getElementById('recommendation');
    const predictionDetails = document.getElementById('predictionDetails');

    // Handle form submission
    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Validate form
        if (!validateForm()) {
            return;
        }
        
        // Show loading state
        const submitBtn = predictionForm.querySelector('.btn-primary');
        const originalText = submitBtn.textContent;
        submitBtn.innerHTML = '<span class="loading"></span> Analyzing...';
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
                displayPredictionResult(result.prediction, predictionData);
                
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
            
        } catch (error) {
            console.error('Prediction error:', error);
            showError('Prediction failed: ' + error.message);
        } finally {
            // Restore button state
            submitBtn.textContent = originalText;
            submitBtn.disabled = false;
        }
    });

    function validateForm() {
        const requiredFields = ['item_id', 'store_id', 'dept_id', 'sell_price', 'prediction_date'];
        let isValid = true;
        
        requiredFields.forEach(fieldName => {
            const field = document.getElementById(fieldName);
            if (!field.value.trim()) {
                field.style.borderColor = '#e53e3e';
                isValid = false;
            } else {
                field.style.borderColor = '#e2e8f0';
            }
        });
        
        return isValid;
    }

    function displayPredictionResult(prediction, inputData) {
        const roundedPrediction = Math.round(prediction * 100) / 100;
        
        // Update prediction value
        predictedValue.textContent = roundedPrediction;
        
        // Update confidence badge
        const confidence = getConfidenceLevel(roundedPrediction);
        confidenceBadge.textContent = confidence.label;
        confidenceBadge.style.background = confidence.color;
        
        // Update business impact
        const impact = getBusinessImpact(roundedPrediction, inputData.sell_price);
        businessImpact.textContent = impact.message;
        recommendation.textContent = impact.recommendation;
        
        // Update prediction details
        updatePredictionDetails(inputData, roundedPrediction);
        
        // Show result with animation
        predictionResult.style.display = 'block';
        predictionResult.scrollIntoView({ 
            behavior: 'smooth',
            block: 'center'
        });
    }

    function getConfidenceLevel(prediction) {
        if (prediction >= 5) {
            return { label: 'High Confidence', color: 'rgba(72, 187, 120, 0.8)' };
        } else if (prediction >= 2) {
            return { label: 'Medium Confidence', color: 'rgba(236, 201, 75, 0.8)' };
        } else {
            return { label: 'Low Confidence', color: 'rgba(245, 101, 101, 0.8)' };
        }
    }

    function getBusinessImpact(prediction, price) {
        const revenue = prediction * price;
        
        if (prediction >= 10) {
            return {
                message: `High demand expected - Revenue potential: $${revenue.toFixed(2)}`,
                recommendation: 'Ensure adequate stock, consider promotional pricing'
            };
        } else if (prediction >= 5) {
            return {
                message: `Moderate demand expected - Revenue potential: $${revenue.toFixed(2)}`,
                recommendation: 'Stock according to prediction, monitor closely'
            };
        } else if (prediction >= 1) {
            return {
                message: `Low demand expected - Revenue potential: $${revenue.toFixed(2)}`,
                recommendation: 'Minimal stocking recommended, avoid overstock'
            };
        } else {
            return {
                message: 'Very low demand expected',
                recommendation: 'Consider discontinuing or promotional pricing'
            };
        }
    }

    function updatePredictionDetails(inputData, prediction) {
        const details = [
            { label: 'Product', value: inputData.item_id },
            { label: 'Store', value: inputData.store_id },
            { label: 'Department', value: inputData.dept_id },
            { label: 'Price', value: `$${inputData.sell_price}` },
            { label: 'Date', value: new Date(inputData.prediction_date).toLocaleDateString() },
            { label: 'Special Event', value: inputData.has_event ? 'Yes' : 'No' }
        ];
        
        predictionDetails.innerHTML = details.map(detail => `
            <div class="detail-item">
                <span class="detail-label">${detail.label}</span>
                <span class="detail-value">${detail.value}</span>
            </div>
        `).join('');
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        errorDiv.style.cssText = `
            background: #fed7d7;
            color: #9b2c2c;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border: 1px solid #feb2b2;
        `;
        
        predictionForm.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
    
    // Initialize form defaults
    function initializeForm() {
        // Set default date to tomorrow
        const dateInput = document.getElementById('prediction_date');
        if (dateInput) {
            const today = new Date();
            const tomorrow = new Date(today);
            tomorrow.setDate(today.getDate() + 1);
            
            dateInput.min = today.toISOString().split('T')[0];
            dateInput.value = tomorrow.toISOString().split('T')[0];
        }
        
        // Set default price
        const priceInput = document.getElementById('sell_price');
        if (priceInput) {
            priceInput.value = '9.99';
        }
    }
    
    // Department and product ID validation
    const deptSelect = document.getElementById('dept_id');
    const itemInput = document.getElementById('item_id');
    
    if (deptSelect && itemInput) {
        deptSelect.addEventListener('change', function() {
            const dept = this.value;
            if (dept && itemInput.value) {
                validateProductDepartmentMatch(itemInput.value, dept);
            }
        });
        
        itemInput.addEventListener('blur', function() {
            const dept = deptSelect.value;
            if (dept && this.value) {
                validateProductDepartmentMatch(this.value, dept);
            }
        });
    }
    
    function validateProductDepartmentMatch(itemId, dept) {
        const prefix = dept.split('_')[0]; // Get HOBBIES, HOUSEHOLD, or FOODS
        if (!itemId.startsWith(prefix)) {
            showWarning(`Product ID should start with "${prefix}" for ${dept} department`);
        }
    }
    
    function showWarning(message) {
        const existingWarning = document.querySelector('.warning-message');
        if (existingWarning) {
            existingWarning.remove();
        }
        
        const warningDiv = document.createElement('div');
        warningDiv.className = 'warning-message';
        warningDiv.textContent = message;
        warningDiv.style.cssText = `
            background: #fef5e7;
            color: #744210;
            padding: 10px;
            border-radius: 6px;
            margin-top: 10px;
            border: 1px solid #f6e05e;
            font-size: 0.9rem;
        `;
        
        predictionForm.insertBefore(warningDiv, predictionForm.querySelector('.form-actions'));
        
        setTimeout(() => {
            warningDiv.remove();
        }, 4000);
    }
    
    // Price input formatting
    const priceInput = document.getElementById('sell_price');
    if (priceInput) {
        priceInput.addEventListener('input', function() {
            let value = parseFloat(this.value);
            if (isNaN(value) || value < 0.01) {
                this.value = '0.01';
            } else if (value > 1000) {
                this.value = '1000.00';
            }
        });
    }
    
    // Initialize everything
    initializeForm();
    
    // Add smooth reveal animations
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

// Clear form function
function clearForm() {
    const form = document.getElementById('predictionForm');
    form.reset();
    
    // Reset styles
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.style.borderColor = '#e2e8f0';
    });
    
    // Hide result
    document.getElementById('predictionResult').style.display = 'none';
    
    // Remove any warning/error messages
    const messages = document.querySelectorAll('.warning-message, .error-message');
    messages.forEach(msg => msg.remove());
    
    // Re-initialize defaults
    const dateInput = document.getElementById('prediction_date');
    if (dateInput) {
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        dateInput.value = tomorrow.toISOString().split('T')[0];
    }
    
    const priceInput = document.getElementById('sell_price');
    if (priceInput) {
        priceInput.value = '9.99';
    }
}

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
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);