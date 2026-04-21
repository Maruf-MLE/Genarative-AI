---
title: CNPI RAG Assistant
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# CNPI Self-RAG Backend

FastAPI backend for Chapainawabganj Polytechnic Institute (CNPI) RAG system.

## API Endpoint

**POST** `/ask`

```json
{
  "question": "আপনার প্রশ্ন এখানে লিখুন"
}
```

## Response

```json
{
  "answer": "উত্তর"
}
```
