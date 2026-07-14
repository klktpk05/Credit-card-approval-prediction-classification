/* ================================================================
   CreditAI — Interactive JavaScript
   ================================================================ */

// ─── PARTICLE BACKGROUND ───
(function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W = canvas.width  = window.innerWidth;
  let H = canvas.height = window.innerHeight;

  const particles = Array.from({ length: 60 }, () => ({
    x: Math.random() * W,
    y: Math.random() * H,
    r: Math.random() * 2 + 0.5,
    vx: (Math.random() - 0.5) * 0.4,
    vy: (Math.random() - 0.5) * 0.4,
    alpha: Math.random() * 0.5 + 0.1,
  }));

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(99,102,241,${p.alpha})`;
      ctx.fill();

      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    });

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(99,102,241,${0.15 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  draw();
  window.addEventListener('resize', () => {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  });
})();


// ─── SLIDER DISPLAYS ───
document.addEventListener('DOMContentLoaded', () => {
  const ageSlider = document.getElementById('age');
  const ageDisplay = document.getElementById('age-display');
  if (ageSlider && ageDisplay) {
    ageSlider.addEventListener('input', () => {
      ageDisplay.textContent = ageSlider.value;
      updateSliderFill(ageSlider);
    });
    updateSliderFill(ageSlider);
  }

  const empSlider = document.getElementById('employment_length');
  const empDisplay = document.getElementById('emp-display');
  if (empSlider && empDisplay) {
    empSlider.addEventListener('input', () => {
      empDisplay.textContent = empSlider.value;
      updateSliderFill(empSlider);
    });
    updateSliderFill(empSlider);
  }
});

function updateSliderFill(slider) {
  const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
  slider.style.background = `linear-gradient(to right, #6366f1 ${pct}%, rgba(255,255,255,0.1) ${pct}%)`;
}


// ─── MULTI-STEP FORM ───
let currentStep = 1;

function nextStep(step) {
  // Validate current step before advancing (basic validation)
  if (step > currentStep) {
    const current = document.querySelector(`.form-step[data-step="${currentStep}"]`);
    const inputs = current ? current.querySelectorAll('input[type="number"]') : [];
    let valid = true;
    inputs.forEach(input => {
      if (input.value === '' || isNaN(input.value)) {
        input.style.borderColor = '#ef4444';
        valid = false;
        setTimeout(() => input.style.borderColor = '', 2000);
      }
    });
    if (!valid) return;
  }

  document.querySelectorAll('.form-step').forEach(s => s.classList.remove('active'));
  const target = document.querySelector(`.form-step[data-step="${step}"]`);
  if (target) target.classList.add('active');

  // Update step indicators
  document.querySelectorAll('.step').forEach(s => {
    const n = parseInt(s.dataset.step);
    s.classList.remove('active', 'done');
    if (n === step) s.classList.add('active');
    if (n < step) s.classList.add('done');
  });

  currentStep = step;
  window.scrollTo({ top: document.getElementById('predict-section').offsetTop - 80, behavior: 'smooth' });
}


// ─── PREDICTION FORM SUBMIT ───
document.getElementById('predict-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();

  const form = e.target;
  const btn  = document.getElementById('predict-btn');
  const btnText = btn.querySelector('.btn-text');
  const btnLoad = btn.querySelector('.btn-loading');

  // Show loading state
  btnText.style.display = 'none';
  btnLoad.style.display = 'flex';
  btn.disabled = true;

  // Collect form data
  const fd = new FormData(form);
  const payload = {
    gender:            fd.get('gender'),
    age:               fd.get('age'),
    marital_status:    fd.get('marital_status'),
    family_members:    fd.get('family_members'),
    dwelling:          fd.get('dwelling'),
    has_car:           fd.get('has_car'),
    has_property:      fd.get('has_property'),
    income:            fd.get('income'),
    employment_status: fd.get('employment_status'),
    employment_length: fd.get('employment_length'),
    education_level:   fd.get('education_level'),
    work_phone:        fd.get('work_phone'),
    phone:             fd.get('phone'),
    email:             fd.get('email'),
  };

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (data.error) {
      showError(data.error);
      return;
    }

    showResult(data);
  } catch (err) {
    showError('Network error: ' + err.message);
  } finally {
    btnText.style.display = 'flex';
    btnLoad.style.display = 'none';
    btn.disabled = false;
  }
});


function showResult(data) {
  const formWrapper  = document.getElementById('predict-form');
  const resultPanel  = document.getElementById('result-panel');
  const stepsDiv     = document.querySelector('.steps-indicator');

  formWrapper.style.display  = 'none';
  if (stepsDiv) stepsDiv.style.display = 'none';
  resultPanel.style.display  = 'block';

  if (data.approved) {
    document.getElementById('result-approved').style.display = 'block';
    document.getElementById('result-rejected').style.display  = 'none';

    const bar   = document.getElementById('confidence-bar');
    const value = document.getElementById('confidence-value');
    const modelEl = document.getElementById('model-name-display');

    setTimeout(() => { bar.style.width = data.confidence + '%'; }, 100);
    value.textContent = data.confidence + '%';
    value.style.color = '#10b981';
    if (modelEl) modelEl.textContent = data.model_used || '—';

    // Update hero card
    const heroCard  = document.getElementById('hero-card');
    const herobadge = document.getElementById('card-status-badge');
    if (heroCard)  heroCard.style.boxShadow  = '0 20px 60px rgba(16,185,129,0.3), 0 0 60px rgba(16,185,129,0.15)';
    if (herobage) herobage.textContent = '✓ Approved';

    // Confetti burst
    confettiBurst();
  } else {
    document.getElementById('result-rejected').style.display = 'block';
    document.getElementById('result-approved').style.display = 'none';

    const bar   = document.getElementById('confidence-bar-reject');
    const value = document.getElementById('confidence-value-reject');
    const modelEl = document.getElementById('model-name-display-reject');

    setTimeout(() => { bar.style.width = data.confidence + '%'; }, 100);
    value.textContent = data.confidence + '%';
    value.style.color = '#ef4444';
    if (modelEl) modelEl.textContent = data.model_used || '—';
  }

  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
}


function showError(msg) {
  const resultPanel = document.getElementById('result-panel');
  const form = document.getElementById('predict-form');
  form.style.display = 'none';
  resultPanel.style.display = 'block';
  resultPanel.innerHTML = `
    <div style="text-align:center;padding:40px">
      <div style="font-size:3rem;margin-bottom:16px">⚠️</div>
      <h3 style="color:#ef4444;font-family:'Outfit',sans-serif;margin-bottom:8px">Prediction Error</h3>
      <p style="color:#94a3b8;margin-bottom:24px">${msg}</p>
      <button class="btn-ghost-sm" onclick="resetForm()">← Try Again</button>
    </div>
  `;
}


function resetForm() {
  const form        = document.getElementById('predict-form');
  const resultPanel = document.getElementById('result-panel');
  const stepsDiv    = document.querySelector('.steps-indicator');

  resultPanel.style.display = 'none';
  form.style.display        = 'block';
  if (stepsDiv) stepsDiv.style.display = 'flex';

  // Reset to step 1
  nextStep(1);
}


// ─── CONFETTI BURST ───
function confettiBurst() {
  const colors = ['#6366f1','#a855f7','#10b981','#06b6d4','#f59e0b','#f1f5f9'];
  for (let i = 0; i < 60; i++) {
    const el = document.createElement('div');
    el.style.cssText = `
      position: fixed;
      left: ${Math.random() * 100}vw;
      top: -10px;
      width: ${Math.random() * 8 + 4}px;
      height: ${Math.random() * 8 + 4}px;
      background: ${colors[Math.floor(Math.random() * colors.length)]};
      border-radius: ${Math.random() > 0.5 ? '50%' : '2px'};
      pointer-events: none;
      z-index: 9999;
      animation: confetti-fall ${Math.random() * 2 + 1.5}s ease-in ${Math.random() * 0.5}s forwards;
    `;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 4000);
  }
}

// Inject confetti animation
const style = document.createElement('style');
style.textContent = `
  @keyframes confetti-fall {
    0%   { transform: translateY(-10px) rotate(0deg); opacity: 1; }
    100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
  }
`;
document.head.appendChild(style);


// ─── INTERSECTION OBSERVER — animate model bars ───
(function observeModelBars() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.width = entry.target.dataset.width || entry.target.style.width;
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.2 });

  document.querySelectorAll('.model-bar').forEach(bar => {
    const targetWidth = bar.style.width;
    bar.style.width = '0%';
    bar.dataset.width = targetWidth;
    observer.observe(bar);
  });
})();


// ─── SMOOTH SCROLL FOR NAV LINKS ───
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', (e) => {
    e.preventDefault();
    const target = document.querySelector(anchor.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});


// ─── NAVBAR SCROLL EFFECT ───
window.addEventListener('scroll', () => {
  const navbar = document.querySelector('.navbar');
  if (navbar) {
    navbar.style.boxShadow = window.scrollY > 50
      ? '0 4px 30px rgba(0,0,0,0.5)'
      : 'none';
  }
});
