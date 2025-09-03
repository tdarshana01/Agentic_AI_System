import re, json
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI

# Load models & index once
Settings.llm = None
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

storage_context = StorageContext.from_defaults(persist_dir="data/storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)
query_engine = index.as_query_engine()


def incident_to_embedding_input(incident: dict) -> str:
    """Convert structured incident dict → text for embeddings."""
    return " | ".join(f"{k}: {v}" for k, v in incident.items() if v is not None)


def analyze_incident_logic(incident_json: dict):
    incident_str = incident_to_embedding_input(incident_json)
    retrieval_result = query_engine.retrieve(incident_str)
    retrieved_text = str(retrieval_result)

    if not retrieval_result:
        return {
            "retrieved_context": retrieved_text,
            "ai_answer": "No relevant past incidents found.",
        }

    context_str = "\n\n".join([node.get_content() for node in retrieval_result])
    num_incidents = len(retrieval_result)

    # LLM Prompt
    template_str = """
    You are an incident analysis assistant. 
    Use ALL retrieved past incidents in <CONTEXT> when analyzing the new incident.

    <CONTEXT>
    {context_str}
    </CONTEXT>

    <NEW_INCIDENT>
    {query_str}
    </NEW_INCIDENT>

    Provide final structured output in JSON only:

    <FINAL_ANSWER_JSON>
    {{
      "incident": "{query_str}",
      "risk_level": "High/Medium/Low",
      "root_causes": ["List causes"],
      "recommended_actions": ["List actions"],
      "relevant_past_incidents_count": {num_incidents}
    }}
    </FINAL_ANSWER_JSON>
    """

    prompt = template_str.format(
        context_str=context_str,
        query_str=incident_str,
        num_incidents=num_incidents,
    )

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    completion = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct-v0.3",
        messages=[
            {"role": "user", "content": "You are a helpful maintenance AI assistant."+ prompt},
        
        ],
        
        temperature=0.3,
    )

    llm_answer = completion.choices[0].message.content

    # --- Extract JSON ---
    m = re.search(r"<FINAL_ANSWER_JSON>\s*(\{.*?\})\s*</FINAL_ANSWER_JSON>", llm_answer, re.DOTALL)
    if m:
        payload = m.group(1)
    else:
        # fallback: try to extract first {...} block
        m2 = re.search(r"\{.*\}", llm_answer, re.DOTALL)
        payload = m2.group(0) if m2 else "{}"

    # Parse JSON safely
    payload = re.sub(r",\s*([}\]])", r"\1", payload)  # remove trailing commas
    try:
        data = json.loads(payload)
    except Exception:
        return {"retrieved_context": retrieved_text, "ai_answer": f"Invalid JSON from LLM: {llm_answer}"}

    # --- Put helper functions here ---
    def normalize_cause(item):
        if isinstance(item, dict):
            return item.get("cause") or item.get("description") or json.dumps(item)
        return str(item).strip()

    def normalize_action(item):
        if isinstance(item, dict):
            return item.get("action") or item.get("description") or json.dumps(item)
        return str(item).strip()

    # Sanitize fields
    incident = data.get("incident", incident_str)
    risk = data.get("risk_level", "Medium").capitalize()
    if risk not in {"High", "Medium", "Low"}:
        risk = "Medium"

    root_causes = [normalize_cause(x) for x in data.get("root_causes", []) if x]
    actions = [normalize_action(x) for x in data.get("recommended_actions", []) if x]
    count = int(data.get("relevant_past_incidents_count", num_incidents))

    # Render final text answer
    machine = incident_json.get("machine", "")
    itype = incident_json.get("type", "")
    status = incident_json.get("status", "")

    final_answer = (
        f"Incident:\n"
        f"- Machine: {machine}\n"
        f"- Type: {itype}\n"
        f"- Status: {status}\n\n"
        f"Risk Level: {risk}\n\n"
        "Root Cause(s):\n"
        f"{chr(10).join(f'- {rc}' for rc in root_causes) if root_causes else '- None found'}\n\n"
        "Recommended Action(s):\n"
        f"{chr(10).join(f'- {ra}' for ra in actions) if actions else '- None found'}\n\n"
        f"Relevant Past Incidents Count: {count}\n"
    )

    return {
        "retrieved_context": retrieved_text,
        "ai_answer": final_answer,
    }
