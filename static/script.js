const form = document.getElementById("churnForm");
const statusText = document.getElementById("statusText");
const predictionText = document.getElementById("predictionText");
const probabilityText = document.getElementById("probabilityText");

form.addEventListener("submit", async function (e) {
  e.preventDefault();

  statusText.textContent = "Loading...";
  predictionText.textContent = "Please wait...";
  probabilityText.textContent = "--";

const data = {
  gender: document.getElementById("gender").value,
  SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),
  Partner: document.getElementById("Partner").value,
  Dependents: document.getElementById("Dependents").value,
  tenure: parseInt(document.getElementById("tenure").value),
  PhoneService: document.getElementById("PhoneService").value,
  MultipleLines: "No",
  InternetService: document.getElementById("InternetService").value,
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: document.getElementById("Contract").value,
  PaperlessBilling: document.getElementById("PaperlessBilling").value,
  PaymentMethod: document.getElementById("PaymentMethod").value,
  MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
  TotalCharges: parseFloat(document.getElementById("TotalCharges").value)
};

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Prediction failed");
    }

    statusText.textContent = "Success";
    predictionText.textContent = result.churn;
    probabilityText.textContent = (result.probability * 100).toFixed(2) + "%";
  } catch (error) {
    statusText.textContent = "Error";
    predictionText.textContent = "Request failed";
    probabilityText.textContent = error.message;
  }
});