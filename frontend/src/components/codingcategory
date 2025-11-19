import React, { useState } from "react";
import Editor from "react-simple-code-editor";
import Prism from "prismjs";
// Import CSS for Prism themes, e.g. "prismjs/themes/prism.css"

export default function CodingModule({ ageGroup }) {
  const [code, setCode] = useState("// Ask the AI for help or paste code here!");
  const [aiResponse, setAiResponse] = useState("");
  const [prompt, setPrompt] = useState("");

  // "Explain", "Generate", or "Edit" code
  const askAI = async (action) => {
    const res = await fetch('/coding', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: `${action}: ${prompt}\nCode:\n${code}`,
        age: ageGroup
      }),
    });
    const { answer } = await res.json();
    setAiResponse(answer);
  };

  return (
    <div>
      <h2>Coding Workspace</h2>
      <textarea
        style={{ width: "100%", marginBottom: 8 }}
        placeholder="What do you want the AI to do? (e.g. explain, fix, generate, etc.)"
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
      />
      <Editor
        value={code}
        onValueChange={setCode}
        highlight={code => Prism.highlight(code, Prism.languages.javascript, 'javascript')}
        padding={10}
        style={{ fontFamily: 'monospace', background: "#fafafa", minHeight: "120px" }}
      />
      <div style={{ marginTop: 8 }}>
        <button onClick={() => askAI('Explain')}>Explain</button>
        <button onClick={() => askAI('Generate')}>Generate</button>
        <button onClick={() => askAI('Edit')}>Edit</button>
      </div>
      {aiResponse && (
        <div style={{ marginTop: 16, whiteSpace: "pre-wrap", background: "#eef", padding: 8 }}>
          <strong>AI Response:</strong>
          <div>{aiResponse}</div>
        </div>
      )}
    </div>
  );
}
