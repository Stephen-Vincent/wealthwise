# wealthwise

# 💼 WealthWise

**WealthWise** is a personal finance and investment planning API built with FastAPI. It helps users onboard with financial goals, assess risk profiles, simulate portfolio outcomes, and receive personalized stock recommendations.

This backend powers features like:

- Risk-based onboarding
- Historical portfolio simulation
- Goal tracking and planning
- Stock recommendations based on risk tolerance

---

## 📘 API Reference

### 🟢 `GET /`

**Description:**  
Health check to confirm the API is live.

**Request:**  
_None_

**Response:**

```json
{ "message": "WealthWise API is running." }
```

---

### 🟢 `POST /onboarding`

**Description:**  
Submits onboarding data, calculates the user's risk score, and stores the submission.

**Request Body:**

```json
{
  "experience": 2,
  "goal": "string",
  "lumpSum": 1000,
  "monthly": 200,
  "timeframe": "1–5 years",
  "consent": true,
  "user_id": 1,
  "income_bracket": "Medium (£30k–£70k)",
  "target_achieved": false,
  "name": "Stephen"
}
```

**Response:**

```json
{
  "id": 5,
  "risk": "Medium",
  "risk_score": 45
}
```

---

### 🟢 `GET /stock-name-map`

**Description:**  
Returns a dictionary mapping stock tickers to company names.

**Response Example:**

```json
{
  "AAPL": "Apple Inc.",
  "GOOG": "Alphabet Inc.",
  "TSLA": "Tesla, Inc."
}
```

---

### 🟢 `POST /simulate-portfolio`

**Description:**  
Generates a portfolio simulation based on a submitted onboarding record.

**Request Body:**

```json
{ "id": 5 } // onboarding submission ID
```

**Response:**

```json
{
  "risk": "Medium",
  "target_value": 15000,
  "simulation_id": 12,
  "portfolio": { ... },
  "timeline": [ ... ]
}
```

---

### 🟢 `POST /recommend-stocks`

**Description:**  
Recommends stocks based on the user's risk profile and timeframe.

**Request Body:**

```json
{
  "risk_score": 45,
  "timeframe": 5
}
```

**Response:**

```json
{
  "recommendations": [
    { "ticker": "AAPL", "volatility": 0.12 },
    { "ticker": "MSFT", "volatility": 0.09 }
  ]
}
```

---

### 🔴 `DELETE /clear-database`

**Description:**  
Deletes all onboarding submissions in the database.

**Response:**

```json
{ "message": "Deleted 10 submissions." }
```

---

### 🔐 `POST /auth/signup`

**Description:**  
Creates a new user.

**Request Body:**

```json
{
  "name": "Stephen",
  "email": "stephen@example.com",
  "password": "password123"
}
```

**Response:**

```json
{
  "message": "User created",
  "user_id": 1
}
```

---

### 🔐 `POST /auth/login`

**Description:**  
Authenticates a user.

**Request Body:**

```json
{
  "email": "stephen@example.com",
  "password": "password123"
}
```

**Response:**

```json
{
  "message": "Login successful",
  "user_id": 1,
  "name": "Stephen"
}
```

---

### 🟢 `GET /users/{user_id}/simulations`

**Description:**  
Fetches all simulations associated with a specific user.

**Path Param:**  
`user_id` — integer

**Response Example:**

```json
[
  {
    "id": 12,
    "user_id": 1,
    "name": "Stephen",
    "goal": "Save for home",
    "risk": "Medium",
    "risk_score": 45,
    "target_value": 15000,
    "created_at": "2025-06-06T10:00:00Z",
    "timeframe": "1–5 years",
    "income_bracket": "Medium (£30k–£70k)",
    "target_achieved": false
  }
]
```

---

### 🟢 `GET /simulations/{simulation_id}`

**Description:**  
Retrieves a specific simulation by its ID.

**Path Param:**  
`simulation_id` — integer

**Response:**

```json
{
  "id": 12,
  "user_id": 1,
  "name": "Stephen",
  "goal": "Save for home",
  "risk": "Medium",
  "risk_score": 45,
  "target_value": 15000,
  "created_at": "2025-06-06T10:00:00Z",
  "portfolio": { ... },
  "timeline": [ ... ]
}
```

---

### 🔴 `DELETE /simulations/{simulation_id}`

**Description:**  
Deletes a simulation by its ID.

**Path Param:**  
`simulation_id` — integer

**Response:**

```json
{
  "message": "Simulation deleted successfully",
  "simulation_id": 12
}
```

---

### 🟢 `POST /simulations`

**Description:**  
Saves a simulation record to the database.

**Request Body:**

```json
{
  "user_id": 1,
  "name": "Stephen",
  "goal": "Save for home",
  "risk": "Medium",
  "risk_score": 45,
  "target_value": 15000,
  "portfolio_json": { ... },
  "timeline_json": [ ... ],
  "submission_id": 5,
  "target_achieved": false,
  "income_bracket": "Medium (£30k–£70k)"
}
```

**Response:**

```json
{
  "message": "Simulation saved",
  "simulation_id": 12
}
```

---
