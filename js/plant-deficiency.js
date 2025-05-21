// 📄 File: js/plant-deficiency.js

function predictImage() {
  const input = document.getElementById('imageUpload');
  const preview = document.getElementById('preview');
  const result = document.getElementById('result');
  const file = input.files[0];

  if (!file) {
    alert('Please upload an image first.');
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);

  result.innerHTML = '⏳ Predicting...';

  const formData = new FormData();
  formData.append('file', file);

  fetch('http://localhost:5000/predict', { // ⛔ Replace with hosted backend URL
    method: 'POST',
    body: formData,
  })
    .then((res) => res.json())
    .then((data) => {
      result.innerHTML = `✅ <strong>${data.prediction}</strong> (${data.confidence}% confidence)`;
    })
    .catch((err) => {
      console.error(err);
      result.innerHTML = '❌ Error occurred during prediction.';
    });
}
