"""
AI Job Application Tracker — Shared LLM Prompts
"""

def build_prompt_status(obs: dict) -> str:
    return (
        f"You are an AI job-application manager.\n"
        f"Application: {obs['company']} — {obs['role']}\n"
        f"Current status: {obs['status']}\n"
        f"Days left: {obs['days_left']}\n\n"
        f"What is the current status of this application?\n"
        f"Respond with ONLY ONE word, no explanation: applied, interview, rejected, or offer"
    )


def build_prompt_priority(obs: dict) -> str:
    return (
        f"You are an AI job-application manager.\n"
        f"Application: {obs['company']} — {obs['role']}\n"
        f"Current status: {obs['status']}\n"
        f"Days left: {obs['days_left']}\n\n"
        f"Rules:\n"
        f"- If status is 'offer' or 'interview' → high\n"
        f"- If status is 'applied' and days_left <= 5 → high\n"
        f"- If status is 'applied' and days_left > 5 → medium\n"
        f"- If status is 'rejected' → low\n\n"
        f"Assign priority. Respond with ONLY ONE word, no explanation: high, medium, or low"
    )


def build_prompt_action(obs: dict) -> str:
    return (
        f"You are an AI job-application manager.\n"
        f"Application: {obs['company']} — {obs['role']}\n"
        f"Current status: {obs['status']}\n"
        f"Days left: {obs['days_left']}\n\n"
        f"Rules:\n"
        f"- If status is 'applied' → follow_up\n"
        f"- If status is 'interview' → prepare_interview\n"
        f"- If status is 'offer' → accept_offer\n"
        f"- If status is 'rejected' → ignore\n\n"
        f"What is the best action? Respond with ONLY ONE word/phrase, no explanation: "
        f"follow_up, prepare_interview, accept_offer, or ignore"
    )
