// ====================== page.js (Part 1/4) ======================
"use client";
import { useState, useEffect, useRef } from "react";
import apiClient from "../lib/apiClient";
import { useDropzone } from "react-dropzone";
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
  Card,
  CardContent,
  LinearProgress,
  IconButton,
  Chip,
  Snackbar,
  Alert,
  Paper,
  Stack,
  TextField,
  createTheme,
  ThemeProvider,
  CssBaseline,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import LogoutIcon from "@mui/icons-material/Logout";
import AddIcon from "@mui/icons-material/Add";
import MicIcon from "@mui/icons-material/Mic";
import JSZip from "jszip";

// ---- THEME ----
const theme = createTheme({
  palette: {
    mode: "light",
    primary:   { main: "#0F2A54" },
    secondary: { main: "#00D1FF" },
    success:   { main: "#00C36F" },
    warning:   { main: "#FFB42A" },
    error:     { main: "#FF3B30" },
    background:{ default: "#F4F6F8", paper: "#FFFFFF" },
    text:      { primary:"#21262E", secondary:"#707A8A" }
  }
});

// ---- AXIOS INTERCEPTORS FOR JWT ----
import axios from "axios";
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("jwt");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem("jwt");
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

function AuthForm({ onAuth }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (mode === "register") {
        await apiClient.post("/auth/register", { email, password });
        setMode("login");
        setError("Registration successful! Please log in.");
      } else {
        const params = new URLSearchParams({ username: email, password });
        const res = await apiClient.post(
          "/auth/jwt/login",
          params,
          { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
        );
        onAuth(res.data.access_token);
      }
    } catch (err) {
      setError(err.response?.data?.detail?.toString() || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="xs">
      <Paper elevation={3} sx={{ p: 4, mt: 8 }}>
        <Typography variant="h5" align="center" gutterBottom>
          {mode === "login" ? "Login" : "Register"}
        </Typography>
        {error && (
          <Alert severity={mode === "register" && error.startsWith("Registration") ? "success" : "error"} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        <form onSubmit={handleSubmit}>
          <TextField
            label="Email"
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
            fullWidth
            sx={{ mb: 2 }}
          />
          <TextField
            label="Password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
            fullWidth
            sx={{ mb: 2 }}
          />
          <Button
            type="submit"
            variant="contained"
            color="primary"
            fullWidth
            disabled={loading}
            sx={{ mb: 2 }}
          >
            {loading
              ? mode === "login"
                ? "Logging in…"
                : "Registering…"
              : mode === "login"
                ? "Log In"
                : "Register"}
          </Button>
        </form>
        <Typography variant="body2" align="center">
          {mode === "login" ? "Need an account?" : "Already have an account?"}{" "}
          <Button
            variant="text"
            size="small"
            onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}
          >
            {mode === "login" ? "Register" : "Login"}
          </Button>
        </Typography>
      </Paper>
    </Container>
  );
}

function RedFlagsManager({ jwt, customRedFlags, setCustomRedFlags, onSave }) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: "", severity: "success" });

  useEffect(() => {
    async function fetchFlags() {
      setLoading(true);
      try {
        const res = await apiClient.get("/red-flags/");
        setCustomRedFlags(res.data.custom_red_flags || []);
      } catch (e) {}
      setLoading(false);
    }
    fetchFlags();
    // eslint-disable-next-line
  }, [jwt]);

  const addFlag = () => {
    if (!input.trim() || customRedFlags.includes(input.trim())) return;
    setCustomRedFlags([...customRedFlags, input.trim()]);
    setInput("");
  };

  const removeFlag = (flag) => {
    setCustomRedFlags(customRedFlags.filter(f => f !== flag));
  };

  const saveFlags = async () => {
    setLoading(true);
    try {
      await apiClient.post("/update-red-flags/", customRedFlags);
      setSnackbar({ open: true, message: "Custom red flags saved!", severity: "success" });
      if (onSave) onSave(customRedFlags);
    } catch (e) {
      setSnackbar({ open: true, message: "Failed to save red flags.", severity: "error" });
    }
    setLoading(false);
  };

  return (
    <Card sx={{ mb: 4 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Custom Red Flags
        </Typography>
        <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: "wrap" }}>
          {customRedFlags.map((flag, idx) => (
            <Chip
              key={flag}
              label={flag}
              onDelete={() => removeFlag(flag)}
              color="error"
              sx={{ mb: 1 }}
            />
          ))}
        </Stack>
        <Box sx={{ display: "flex", gap: 1 }}>
          <TextField
            label="Add new red flag"
            value={input}
            onChange={e => setInput(e.target.value)}
            size="small"
            fullWidth
            onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); addFlag(); } }}
          />
          <IconButton color="primary" onClick={addFlag} aria-label="add">
            <AddIcon />
          </IconButton>
          <Button
            onClick={saveFlags}
            variant="contained"
            color="primary"
            disabled={loading}
          >
            Save
          </Button>
        </Box>
      </CardContent>
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Card>
  );
}

// Helper to determine button color based on call score
function getCallScoreColor(overall_perf) {
  const score = parseInt((overall_perf || "").split("/")[0], 10);
  if (isNaN(score)) return "primary";
  if (score >= 7) return "success";
  if (score >= 4) return "warning";
  return "error";
}

// Helper to group downloads by representative name
function groupByRep(downloads) {
  return downloads.reduce((acc, dl) => {
    const rep = dl.rep_name || "Unknown Rep";
    if (!acc[rep]) acc[rep] = [];
    acc[rep].push(dl);
    return acc;
  }, {});
}

// RecordCallButton component
function RecordCallButton({ onUpload }) {
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);
    chunksRef.current = [];
    mediaRecorderRef.current.ondataavailable = e => chunksRef.current.push(e.data);
    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/wav" });
      const file = new File([blob], `recording_${Date.now()}.wav`, { type: "audio/wav" });
      onUpload([file]);
    };
    mediaRecorderRef.current.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  return (
    <Button
      variant="contained"
      color="secondary"
      sx={{ mt: 2 }}
      startIcon={<MicIcon />}
      onClick={recording ? stopRecording : startRecording}
    >
      {recording ? "Stop Recording" : "Record a call"}
    </Button>
  );
}

// --- Rep Stats Panel ---
function RepStatsPanel({ jwt }) {
  const [stats, setStats] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!jwt) return;
    setLoading(true);
    apiClient.get("/rep-stats/")
      .then(res => setStats(res.data.rep_stats || []))
      .catch(() => setStats([]))
      .finally(() => setLoading(false));
  }, [jwt]);

  const safeStats = Array.isArray(stats) ? stats : Object.values(stats || {});

  return (
    <Card sx={{ mb: 4 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Rep Stats
        </Typography>
        {loading ? (
          <LinearProgress />
        ) : safeStats.length === 0 ? (
          <Typography variant="body2" color="text.secondary">No stats available.</Typography>
        ) : (
          safeStats.map(stat => (
            <Box key={stat.rep_name} sx={{ mb: 2 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                {stat.rep_name}
              </Typography>
              <Stack direction="row" spacing={2} sx={{ mb: 1, flexWrap: "wrap" }}>
                <Chip label={`Avg. Score: ${stat.average_performance}`} color="primary" />
                <Chip label={`Talk:Listen: ${stat.talk_to_listen_ratio ?? "N/A"}`} color="secondary" />
                <Chip label={`Avg. 'No's: ${stat.average_nos_before_accept}`} color="warning" />
                <Chip label={`Calls: ${stat.calls}`} color="success" />
              </Stack>
            </Box>
          ))
        )}
      </CardContent>
    </Card>
  );
}

// --- Coaching Plan Modal ---
function CoachingPlanModal({ open, onClose, repName, jwt }) {
  const [plan, setPlan] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open || !repName) return;
    setPlan("");
    setError("");
    setLoading(true);
    apiClient.post(`/coaching-plan/${encodeURIComponent(repName)}`)
      .then(res => setPlan(res.data.coaching_plan))
      .catch(e => setError("Failed to fetch coaching plan."))
      .finally(() => setLoading(false));
  }, [open, repName, jwt]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Coaching Plan for {repName}</DialogTitle>
      <DialogContent dividers>
        {loading ? (
          <LinearProgress />
        ) : error ? (
          <Alert severity="error">{error}</Alert>
        ) : (
          <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
            {plan}
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

// ====================== END OF PART 1/4 ======================
// ====================== page.js (Part 2/4) ======================

export default function Home() {
  const [jwt, setJwt] = useState(null);
  const [error, setError] = useState("");
  const [downloads, setDownloads] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [completedAnalyses, setCompletedAnalyses] = useState(0);
  const [totalFiles, setTotalFiles] = useState(0);
  const [status, setStatus] = useState("idle");
  const [minutes, setMinutes] = useState(null);
  const [customRedFlags, setCustomRedFlags] = useState([]);
  const [buyDialogOpen, setBuyDialogOpen] = useState(false);
  const [buyAmount, setBuyAmount] = useState(60);
  const [buyLoading, setBuyLoading] = useState(false);
  const [buyError, setBuyError] = useState("");
  const [filePolling, setFilePolling] = useState({});
  const [coachingRep, setCoachingRep] = useState("");
  const [coachingOpen, setCoachingOpen] = useState(false);

  // --- PERSIST analyses in localStorage ---
  useEffect(() => {
    const saved = localStorage.getItem("downloads");
    if (saved) setDownloads(JSON.parse(saved));
  }, []);

  useEffect(() => {
    localStorage.setItem("downloads", JSON.stringify(downloads));
  }, [downloads]);

  useEffect(() => {
    const token = localStorage.getItem("jwt");
    if (token) setJwt(token);
  }, []);

  useEffect(() => {
    if (jwt) {
      apiClient.get("/users/me").then(res => setMinutes(res.data.minutes));
    }
  }, [jwt, completedAnalyses]);

  // --- NEW: Resume polling for in-progress jobs after refresh ---
  useEffect(() => {
    const inProgress = JSON.parse(localStorage.getItem("inProgressJobs") || "[]");
    const savedTotalFiles = localStorage.getItem("totalFiles");
    const savedCompletedAnalyses = localStorage.getItem("completedAnalyses");
    if (savedTotalFiles) setTotalFiles(Number(savedTotalFiles));
    if (savedCompletedAnalyses) setCompletedAnalyses(Number(savedCompletedAnalyses));
    if (inProgress.length > 0) {
      setFilePolling(prev => {
        const updated = { ...prev };
        inProgress.forEach(job => {
          if (!updated[job.job_id]) {
            updated[job.job_id] = {
              pollingMessage: `Restoring analysis for "${job.name}"...`,
              error: "",
              status: "pending",
              name: job.name
            };
          }
        });
        return updated;
      });
    }
    inProgress.forEach(job => {
      pollJobStatus(job.job_id, null, job.name);
    });
  }, []);

  useEffect(() => {
    localStorage.setItem("totalFiles", totalFiles);
  }, [totalFiles]);
  useEffect(() => {
    localStorage.setItem("completedAnalyses", completedAnalyses);
  }, [completedAnalyses]);
  useEffect(() => {
    localStorage.setItem("filePolling", JSON.stringify(filePolling));
  }, [filePolling]);
  useEffect(() => {
    const savedFilePolling = localStorage.getItem("filePolling");
    if (savedFilePolling) setFilePolling(JSON.parse(savedFilePolling));
  }, []);

  const handleAuth = (token) => {
    localStorage.setItem("jwt", token);
    setJwt(token);
  };

  const handleLogout = () => {
    localStorage.removeItem("jwt");
    setJwt(null);
    setMinutes(null);
    setDownloads([]);
    localStorage.removeItem("downloads");
    localStorage.removeItem("inProgressJobs");
    localStorage.removeItem("totalFiles");
    localStorage.removeItem("completedAnalyses");
    localStorage.removeItem("filePolling");
  };

  const handleStopAnalyzing = () => {
    setFilePolling({});
    localStorage.removeItem("inProgressJobs");
    setCompletedAnalyses(totalFiles);
    setError("⏹️ Analysis stopped by user. You may re-upload files to analyze again.");
  };

  // --- IMPROVED PER-FILE POLLING LOGIC ---
  const pollJobStatus = (job_id, file, name) => {
    let attempts = 0;
    let maxAttempts = 120;
    const poll = async () => {
      try {
        const res = await apiClient.get(`/job-status/${job_id}`);
        console.log("Job status response:", res.data); // Debug log for backend response
        if (res.data.status === "done" || res.data.status === "completed") { // Accept both status values
          setFilePolling(prev => ({
            ...prev,
            [job_id]: { ...prev[job_id], pollingMessage: "", status: "done" }
          }));
          const inProgress = JSON.parse(localStorage.getItem("inProgressJobs") || "[]");
          const updated = inProgress.filter(j => j.job_id !== job_id);
          localStorage.setItem("inProgressJobs", JSON.stringify(updated));

          const d = res.data.result;
          if (
            !d ||
            !d.full_analysis_content ||
            !d.overall_perf ||
            !d.rep_name ||
            !d.categorization
          ) {
            setFilePolling(prev => ({
              ...prev,
              [job_id]: {
                ...prev[job_id],
                error: `❌ Error processing ${name}: No analysis returned from server.`,
                status: "error",
                pollingMessage: ""
              }
            }));
            setCompletedAnalyses((prev) => prev + 1);
            return;
          }

          const {
            full_analysis_content,
            overall_perf,
            rep_name,
            categorization,
            red_flags,
            red_flag_reason,
            red_flag_quotes,
            transcript,
            title
          } = d;

          const filename = title || `${overall_perf} ${rep_name} ${categorization} ${name}.html`;

          const blob = new Blob(
            [
              `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Analysis Report: ${filename}</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; background-color: #F4F6F8; color: #21262E; padding: 20px; }
    .container { max-width: 900px; margin: auto; padding: 25px; background-color: #FFFFFF; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-radius: 8px; }
    h1, h2 { color: #00D1FF; }
    .analysis-content { line-height: 1.6; font-size: 16px; margin-bottom: 20px; }
    .red-flags { background-color: #FFEBE9; border-left: 5px solid #FF3B30; padding: 15px; border-radius: 5px; margin-top: 20px; }
    blockquote { font-style: italic; background-color: #E9EEF2; padding: 10px; border-radius: 5px; }
    .transcript-block { background: #E9EEF2; padding: 16px; border-radius: 6px; white-space: pre-wrap; font-family: monospace; margin-top: 2em; }
    footer { margin-top: 30px; text-align: center; color: #707A8A; font-size: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>📊 Analysis Report: ${filename}</h1>
    <div class="analysis-content">
      ${full_analysis_content.replace(/\n/g, "<br />")}
    </div>
    ${
      red_flags === "Yes"
        ? `<div class="red-flags">
      <h2>🚩 Red Flags Detected</h2>
      <p><strong>Reason:</strong> ${
        red_flag_reason || "N/A"
      }</p>
      ${
        red_flag_quotes !== "None"
          ? `<blockquote>${red_flag_quotes}</blockquote>`
          : ""
      }
    </div>`
        : ""
    }
    <hr style="margin:2em 0;">
    <h2>📜 Call Transcript</h2>
    <div class="transcript-block">
${transcript ? transcript : "No transcript available."}
    </div>
    <footer>Generated automatically by Call Analyzer AI.</footer>
  </div>
</body>
</html>`
            ],
            { type: "text/html" }
          );

          const url = URL.createObjectURL(blob);

          setDownloads(prev => {
            const newDownloads = [
              ...prev,
              {
                name: filename,
                url,
                red_flags,
                red_flag_reason,
                red_flag_quotes,
                overall_perf,
                color: getCallScoreColor(overall_perf),
                rep_name,
              }
            ];
            newDownloads.sort((a, b) => {
              if ((a.rep_name || "") < (b.rep_name || "")) return -1;
              if ((a.rep_name || "") > (b.rep_name || "")) return 1;
              const aScore = parseInt(a.overall_perf?.split("/")[0] || "0", 10);
              const bScore = parseInt(b.overall_perf?.split("/")[0] || "0", 10);
              return bScore - aScore;
            });
            return newDownloads;
          });
          setCompletedAnalyses((prev) => prev + 1);
        } else if (res.data.status === "error") {
          const inProgress = JSON.parse(localStorage.getItem("inProgressJobs") || "[]");
          const updated = inProgress.filter(j => j.job_id !== job_id);
          localStorage.setItem("inProgressJobs", JSON.stringify(updated));

          setFilePolling(prev => ({
            ...prev,
            [job_id]: {
              ...prev[job_id],
              pollingMessage: "",
              error: `❌ Error processing ${name}: ${res.data.result?.error || "Unknown backend error."}`,
              status: "error"
            }
          }));
          setCompletedAnalyses((prev) => prev + 1);
        } else {
          if (++attempts < maxAttempts) {
            setTimeout(poll, 5000);
            setFilePolling(prev => ({
              ...prev,
              [job_id]: {
                ...prev[job_id],
                pollingMessage: `Still processing "${name}"... (this may take a few minutes for large files)`,
                status: res.data.status
              }
            }));
          } else {
            const inProgress = JSON.parse(localStorage.getItem("inProgressJobs") || "[]");
            const updated = inProgress.filter(j => j.job_id !== job_id);
            localStorage.setItem("inProgressJobs", JSON.stringify(updated));

            setFilePolling(prev => ({
              ...prev,
              [job_id]: {
                ...prev[job_id],
                pollingMessage: "",
                error: `❌ Timeout waiting for analysis of ${name}.`,
                status: "error"
              }
            }));
            setCompletedAnalyses((prev) => prev + 1);
          }
        }
      } catch (err) {
        if (++attempts < maxAttempts) {
          setTimeout(poll, 5000);
          setFilePolling(prev => ({
            ...prev,
            [job_id]: {
              ...prev[job_id],
              pollingMessage: `Still processing "${name}"... (waiting for server, no response yet; this is normal for large files)`,
              status: "pending"
            }
          }));
        } else {
          const inProgress = JSON.parse(localStorage.getItem("inProgressJobs") || "[]");
          const updated = inProgress.filter(j => j.job_id !== job_id);
          localStorage.setItem("inProgressJobs", JSON.stringify(updated));

          setFilePolling(prev => ({
            ...prev,
            [job_id]: {
              ...prev[job_id],
              pollingMessage: "",
              error: `❌ Network error or timeout polling job status for "${name}".`,
              status: "error"
            }
          }));
          setCompletedAnalyses((prev) => prev + 1);
        }
      }
    };
    poll();
  };

  // ====================== END OF PART 2/4 ======================
// ====================== page.js (Part 3/4) ======================

// --- Upload Handler ---
const handleDrop = async (acceptedFiles) => {
  setError("");
  if (!acceptedFiles.length) return;
  if (!minutes || isNaN(minutes)) {
    setError("Unable to retrieve your available minutes. Please refresh or log in again.");
    return;
  }

  // Estimate total minutes required
  let totalMinutesRequired = 0;
  for (const file of acceptedFiles) {
    // Estimate using file size (not perfect, but a proxy: 1MB ~ 1 min for mp3, ~10MB ~ 1 min for wav)
    const ext = file.name.split(".").pop().toLowerCase();
    let est = 1;
    if (ext === "mp3" || ext === "m4a" || ext === "aac") {
      est = Math.max(1, Math.round(file.size / 1_000_000));
    } else if (ext === "wav" || ext === "flac") {
      est = Math.max(1, Math.round(file.size / 10_000_000));
    }
    totalMinutesRequired += est;
  }

  if (minutes < totalMinutesRequired) {
    setError(
      `You have only ${minutes} minute${minutes === 1 ? "" : "s"} left, but these files may require about ${totalMinutesRequired} minute${totalMinutesRequired === 1 ? "" : "s"} to analyze. Please buy more minutes.`
    );
    return;
  }

  setTotalFiles(acceptedFiles.length);
  setCompletedAnalyses(0);
  setStatus("analyzing");

  let inProgressJobs = [];
  for (const file of acceptedFiles) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("custom_red_flags", JSON.stringify(customRedFlags));
    try {
      setFilePolling(prev => ({
        ...prev,
        [file.name]: { pollingMessage: `Uploading "${file.name}"...`, error: "", status: "uploading", name: file.name }
      }));
      const res = await apiClient.post("/upload/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percent);
        }
      });
      setFilePolling(prev => ({
        ...prev,
        [res.data.job_id]: { pollingMessage: `Analyzing "${file.name}"...`, error: "", status: "pending", name: file.name }
      }));
      inProgressJobs.push({ job_id: res.data.job_id, name: file.name });
      pollJobStatus(res.data.job_id, file, file.name);
    } catch (err) {
      setFilePolling(prev => ({
        ...prev,
        [file.name]: {
          pollingMessage: "",
          error: `❌ Error uploading "${file.name}": ${err.response?.data?.detail || err.message}`,
          status: "error",
          name: file.name
        }
      }));
      setCompletedAnalyses((prev) => prev + 1);
    }
  }
  localStorage.setItem("inProgressJobs", JSON.stringify(inProgressJobs));
  setStatus("idle");
};

const { getRootProps, getInputProps, isDragActive } = useDropzone({
  onDrop: handleDrop,
  accept: {
    "audio/*": [".mp3", ".wav", ".m4a", ".aac", ".flac"]
  },
  multiple: true
});

// --- Buy Minutes Dialog ---
const handleBuyMinutes = async () => {
  setBuyLoading(true);
  setBuyError("");
  try {
    // Simulate payment and add minutes
    await apiClient.post("/buy-minutes/", { minutes: buyAmount });
    setBuyDialogOpen(false);
    setMinutes((prev) => Number(prev) + Number(buyAmount));
  } catch (e) {
    setBuyError(e.response?.data?.detail || "Failed to buy minutes.");
  }
  setBuyLoading(false);
};

// --- Download All as ZIP ---
const handleDownloadAll = async () => {
  const zip = new JSZip();
  downloads.forEach(dl => {
    zip.file(dl.name, fetch(dl.url).then(r => r.blob()));
  });
  const content = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(content);
  const a = document.createElement("a");
  a.href = url;
  a.download = "call-analyses.zip";
  a.click();
  URL.revokeObjectURL(url);
};

// --- UI ---
return (
  <ThemeProvider theme={theme}>
    <CssBaseline />
    <AppBar position="static" color="primary" elevation={2}>
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Call Analyzer AI
        </Typography>
        {jwt && (
          <>
            <Chip
              label={
                minutes === null
                  ? "Loading minutes…"
                  : `${minutes} minute${minutes === 1 ? "" : "s"} left`
              }
              color={minutes > 0 ? "success" : "warning"}
              sx={{ mr: 2 }}
              onClick={() => setBuyDialogOpen(true)}
              clickable
            />
            <Button
              color="secondary"
              variant="outlined"
              onClick={() => setBuyDialogOpen(true)}
              sx={{ mr: 2 }}
            >
              Buy Minutes
            </Button>
            <IconButton color="inherit" onClick={handleLogout}>
              <LogoutIcon />
            </IconButton>
          </>
        )}
      </Toolbar>
    </AppBar>
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      {!jwt ? (
        <AuthForm onAuth={handleAuth} />
      ) : (
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <RedFlagsManager
              jwt={jwt}
              customRedFlags={customRedFlags}
              setCustomRedFlags={setCustomRedFlags}
            />
            <RepStatsPanel jwt={jwt} />
          </Grid>
          <Grid item xs={12} md={8}>
            <Paper
              elevation={3}
              sx={{
                p: 4,
                mb: 3,
                borderRadius: 3,
                background: "#F7FBFD"
              }}
            >
              <Typography variant="h5" gutterBottom>
                Upload Call Recordings
              </Typography>
              <Box
                {...getRootProps()}
                sx={{
                  border: "2px dashed #00D1FF",
                  borderRadius: 2,
                  p: 3,
                  textAlign: "center",
                  mb: 2,
                  background: isDragActive ? "#E0F7FF" : "#F7FBFD",
                  cursor: "pointer"
                }}
              >
                <input {...getInputProps()} />
                <CloudUploadIcon sx={{ fontSize: 48, color: "#00D1FF" }} />
                <Typography variant="body1" sx={{ mt: 1 }}>
                  {isDragActive
                    ? "Drop the files here…"
                    : "Drag & drop audio files here, or click to select files"}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Supported: .mp3, .wav, .m4a, .aac, .flac. Each file ≤ 2 hours.
                </Typography>
              </Box>
              <RecordCallButton onUpload={handleDrop} />
              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}
              {status === "analyzing" && (
                <Box sx={{ mt: 3 }}>
                  <LinearProgress variant="determinate" value={uploadProgress} />
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Uploading… {uploadProgress}%
                  </Typography>
                </Box>
              )}
              {Object.keys(filePolling).length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Analysis Progress
                  </Typography>
                  {Object.entries(filePolling).map(([job_id, poll]) => (
                    <Box key={job_id} sx={{ mb: 2 }}>
                      <Typography variant="subtitle2">{poll.name}</Typography>
                      {poll.pollingMessage && (
                        <Typography variant="body2" color="text.secondary">
                          {poll.pollingMessage}
                        </Typography>
                      )}
                      {poll.error && (
                        <Alert severity="error" sx={{ mt: 1 }}>
                          {poll.error}
                        </Alert>
                      )}
                      {poll.status === "done" && (
                        <Alert severity="success" sx={{ mt: 1 }}>
                          Analysis complete!
                        </Alert>
                      )}
                    </Box>
                  ))}
                  <Button
                    variant="outlined"
                    color="error"
                    sx={{ mt: 2 }}
                    onClick={handleStopAnalyzing}
                  >
                    Stop Analyzing
                  </Button>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    {completedAnalyses}/{totalFiles} files analyzed.
                  </Typography>
                </Box>
              )}
            </Paper>
            <Paper elevation={2} sx={{ p: 3, borderRadius: 3 }}>
              <Typography variant="h6" gutterBottom>
                Download Reports
              </Typography>
              {downloads.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No analysis reports yet.
                </Typography>
              ) : (
                <>
                  <Button
                    variant="outlined"
                    color="primary"
                    onClick={handleDownloadAll}
                    sx={{ mb: 2 }}
                  >
                    Download All as ZIP
                  </Button>
                <Button
  variant="outlined"
  color="error"
  onClick={() => {
    setDownloads([]);
    localStorage.removeItem("downloads");
  }}
  sx={{ mb: 2, ml: 2 }}
>
  Clear Calls
</Button>
                  <Divider sx={{ mb: 2 }} />
                  {Object.entries(groupByRep(downloads)).map(([rep, files]) => (
                    <Box key={rep} sx={{ mb: 3 }}>
                      <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 1 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                          {rep}
                        </Typography>
                        <Button
                          variant="outlined"
                          color="secondary"
                          size="small"
                          onClick={() => { setCoachingRep(rep); setCoachingOpen(true); }}
                        >
                          Generate Coaching Plan
                        </Button>
                      </Stack>
                      <Grid container spacing={2}>
                        {files.map((dl, idx) => (
                          <Grid item xs={12} md={6} key={dl.name + idx}>
                            <Card
                              variant="outlined"
                              sx={{
                                mb: 2,
                                borderLeft: `6px solid ${theme.palette[dl.color]?.main || "#0F2A54"}`
                              }}
                            >
                              <CardContent>
                                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                  <strong>{dl.name}</strong>
                                </Typography>
                                <Stack direction="row" spacing={1} sx={{ mb: 1, flexWrap: "wrap" }}>
                                  <Chip label={dl.overall_perf || "N/A"} color={dl.color} />
                                  <Chip label={dl.red_flags === "Yes" ? "Red Flags" : "No Red Flags"} color={dl.red_flags === "Yes" ? "error" : "success"} />
                                  <Chip label={dl.rep_name || "Unknown Rep"} />
                                </Stack>
                                {dl.red_flags === "Yes" && (
                                  <Alert severity="error" sx={{ mb: 1 }}>
                                    {dl.red_flag_reason}
                                  </Alert>
                                )}
                                <Button
                                  variant="contained"
                                  color="primary"
                                  href={dl.url}
                                  download={dl.name}
                                  sx={{ mr: 2 }}
                                >
                                  Download Report
                                </Button>
                              </CardContent>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    </Box>
                  ))}
                </>
              )}
            </Paper>
          </Grid>
        </Grid>
      )}
    </Container>
    {/* Buy Minutes Dialog */}
    <Dialog open={buyDialogOpen} onClose={() => setBuyDialogOpen(false)}>
      <DialogTitle>Buy Minutes</DialogTitle>
      <DialogContent>
        <Typography variant="body2" sx={{ mb: 2 }}>
          $2 per hour of processed talk time (~$0.033/minute). Minutes never expire.
        </Typography>
        <TextField
          label="Minutes to buy"
          type="number"
          value={buyAmount}
          onChange={e => setBuyAmount(Number(e.target.value))}
          fullWidth
          sx={{ mb: 2 }}
          inputProps={{ min: 1 }}
        />
        {buyError && <Alert severity="error">{buyError}</Alert>}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setBuyDialogOpen(false)}>Cancel</Button>
        <Button
          onClick={handleBuyMinutes}
          disabled={buyLoading}
          variant="contained"
          color="primary"
        >
          {buyLoading ? "Processing…" : "Buy"}
        </Button>
      </DialogActions>
    </Dialog>
    {/* Coaching Plan Modal */}
    <CoachingPlanModal
      open={coachingOpen}
      onClose={() => setCoachingOpen(false)}
      repName={coachingRep}
      jwt={jwt}
    />
  </ThemeProvider>
);
}
