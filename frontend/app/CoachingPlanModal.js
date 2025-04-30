import React from "react";
import { Dialog, DialogTitle, DialogContent, DialogActions, Button, TextField, CircularProgress, Alert, Typography } from "@mui/material";

export default function CoachingPlanModal({ open, prompt, setPrompt, onClose, onGenerate, result, loading, error }) {
  // --- Download Coaching Plan ---
  const handleDownload = () => {
    if (!result) return;
    const blob = new Blob([result], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "coaching_plan.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  // --- Upload Coaching Plan (from file) ---
  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setPrompt(text);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Generate Coaching Plan</DialogTitle>
      <DialogContent>
        <Typography variant="body2" sx={{ mb: 1 }}>
          Enter a prompt or context for the coaching plan:
        </Typography>
        <TextField
          label="Prompt"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          fullWidth
          multiline
          minRows={2}
          disabled={loading}
          sx={{ mb: 2 }}
        />
        <Button component="label" variant="outlined" sx={{ mb: 2 }}>
          Upload Prompt
          <input type="file" accept=".txt" hidden onChange={handleUpload} />
        </Button>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        {loading && <CircularProgress size={32} sx={{ display: "block", mx: "auto", my: 2 }} />}
        {result && (
          <Alert severity="success" sx={{ whiteSpace: "pre-line", mt: 2 }}>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>Coaching Plan:</Typography>
            {result}
            <Button sx={{ mt: 2 }} variant="contained" color="secondary" onClick={handleDownload}>
              Download Coaching Plan
            </Button>
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>Close</Button>
        <Button onClick={onGenerate} disabled={loading || !prompt} variant="contained" color="primary">
          Generate
        </Button>
      </DialogActions>
    </Dialog>
  );
}
