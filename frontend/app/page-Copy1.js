"use client";
import { useState, useEffect } from "react";
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
  Grid
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import LogoutIcon from "@mui/icons-material/Logout";
import AddIcon from "@mui/icons-material/Add";
import CoachingPlanModal from "./CoachingPlanModal";

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

// ---- DEBUG LOGGING UTILITY ----
function debugLog(...args) {
  if (window.DEBUG_LOGS || localStorage.getItem('DEBUG_LOGS') === 'true') {
    console.log('[DEBUG]', ...args);
  }
}

// ---- AXIOS INTERCEPTORS FOR JWT ----
import axios from "axios";
apiClient.interceptors.request.use(
  (config) => {
    debugLog('API Request:', config.method, config.url, config);
    const token = localStorage.getItem("jwt");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => { debugLog('API Request Error:', error); return Promise.reject(error); }
);

apiClient.interceptors.response.use(
  (response) => { debugLog('API Response:', response.config.url, response); return response; },
  (error) => {
    debugLog('API Response Error:', error.config?.url, error);
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
    debugLog('AuthForm Submit:', mode, email);
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
        debugLog('Login Success:', res.data);
        onAuth(res.data.access_token);
      }
    } catch (err) {
      debugLog('Auth Error:', err);
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
          <Alert
            severity={
              mode === "register" && error && error.startsWith("Registration")
                ? "success"
                : "error"
            }
            sx={{ mb: 2 }}
          >
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
      debugLog('Fetching custom red flags...');
      try {
        const res = await apiClient.get("/red-flags/");
        debugLog('Fetched red flags:', res.data);
        setCustomRedFlags(res.data.custom_red_flags || []);
      } catch (e) {
        debugLog('Error fetching red flags:', e);
      }
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
    debugLog('Saving custom red flags:', customRedFlags);
    try {
      await apiClient.post("/update-red-flags/", customRedFlags);
      setSnackbar({ open: true, message: "Custom red flags saved!", severity: "success" });
      if (onSave) onSave(customRedFlags);
      debugLog('Red flags saved!');
    } catch (e) {
      debugLog('Failed to save red flags:', e);
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

// Helper to group downloads by normalized representative name
function groupByRep(downloads) {
  return downloads.reduce((acc, dl) => {
    const rep = (dl.rep_name || "Unknown Rep").trim().toLowerCase();
    if (!acc[rep]) acc[rep] = [];
    acc[rep].push(dl);
    return acc;
  }, {});
}

// Helper: Average of a field
function avgField(arr, field) {
  const vals = arr.map(x => typeof x[field] === 'number' ? x[field] : parseFloat(x[field])).filter(v => !isNaN(v));
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

// ---- ANALYSIS HTML TEMPLATE ----
function buildAnalysisHtml({ filename, full_analysis_content, red_flags, red_flag_reason, red_flag_quotes }) {
  return `<!DOCTYPE html>
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
      <p><strong>Reason:</strong> ${red_flag_reason || "N/A"}</p>
      ${
        red_flag_quotes !== "None"
          ? `<blockquote>${red_flag_quotes}</blockquote>`
          : ""
      }
    </div>`
        : ""
    }
    <hr style="margin:2em 0;">
    <footer>Generated automatically by Greener Living AI Analysis.</footer>
  </div>
</body>
</html>`;
}

// ---- PERSISTED DOWNLOADS UTILS ----
function saveDownloadsToStorage(downloads) {
  // Only persist necessary fields (not Blob or url)
  const safeDownloads = downloads.map(dl => ({
    ...dl,
    content: dl.full_analysis_content || dl.content || '',
    // Remove blob and url if present
    blob: undefined,
    url: undefined,
  }));
  localStorage.setItem('downloads', JSON.stringify(safeDownloads));
}

function loadDownloadsFromStorage() {
  try {
    const raw = localStorage.getItem('downloads');
    if (!raw) return [];
    const arr = JSON.parse(raw);
    // Recreate blob and url for each
    return arr.map(dl => {
      const blob = new Blob([dl.content], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      return {
        ...dl,
        blob,
        url,
        full_analysis_content: dl.content,
      };
    });
  } catch {
    return [];
  }
}

export default function Home() {
  const [jwt, setJwt] = useState(null);
  const [error, setError] = useState("");
  const [downloads, setDownloads] = useState(() => loadDownloadsFromStorage());
  const [uploadProgress, setUploadProgress] = useState(0);
  const [completedAnalyses, setCompletedAnalyses] = useState(0);
  const [totalFiles, setTotalFiles] = useState(0);
  const [status, setStatus] = useState("idle");
  const [minutes, setMinutes] = useState(null);
  const [customRedFlags, setCustomRedFlags] = useState([]);
  const [buyDialogOpen, setBuyDialogOpen] = useState(false);
  const [selectedPackage, setSelectedPackage] = useState("starter");
  const [stopAnalysing, setStopAnalysing] = useState(false);
  const [jobProgress, setJobProgress] = useState({});
  const [coachingOpen, setCoachingOpen] = useState(false);
  const [coachingPrompt, setCoachingPrompt] = useState("");
  const [coachingResult, setCoachingResult] = useState("");
  const [coachingLoading, setCoachingLoading] = useState(false);
  const [coachingError, setCoachingError] = useState("");
  const [coachingAnalyses, setCoachingAnalyses] = useState([]);
  const [coachingRep, setCoachingRep] = useState("");
  const [jobStatusMessage, setJobStatusMessage] = useState("");

  // ---- ENV VARS ----
  const COMPANY = process.env.NEXT_PUBLIC_COMPANY || "Greener Living";

  // --- Rep stats calculation ---
  const repStats = groupByRep(downloads);

  // --- Clear Calls ---
  const handleClearCalls = () => {
    setDownloads([]);
    setCompletedAnalyses(0);
    setTotalFiles(0);
    setError("");
    setUploadProgress(0);
  };

  // --- Stop Analysing ---
  const handleStopAnalysing = () => {
    setStopAnalysing(true);
    setStatus("idle");
  };

  useEffect(() => {
    const token = localStorage.getItem("jwt");
    if (token) setJwt(token);
  }, []);

  useEffect(() => {
    if (jwt) {
      apiClient.get("/users/me").then(res => setMinutes(res.data.minutes));
    }
  }, [jwt, completedAnalyses]);

  useEffect(() => {
    saveDownloadsToStorage(downloads);
  }, [downloads]);

  const handleAuth = (token) => {
    debugLog('Auth Success:', token);
    localStorage.setItem("jwt", token);
    setJwt(token);
  };

  const handleLogout = () => {
    debugLog('Logout');
    localStorage.removeItem("jwt");
    setJwt(null);
    setMinutes(null);
    setDownloads([]);
  };

  const pollJobStatus = async (jobId, file, resolve, reject, pollCount = 0, networkRetries = 0) => {
    if (stopAnalysing) {
      reject(new Error("Analysis stopped by user."));
      return;
    }
    try {
      const res = await apiClient.get(`/job-status/${jobId}`);
      const { status: jobStatus, result, message } = res.data;
      setJobProgress(prev => ({ ...prev, [jobId]: { status: jobStatus, result, message } }));
      if (jobStatus === "done") {
        setJobStatusMessage("");
        resolve(result);
      } else if (jobStatus === "error") {
        setJobStatusMessage("");
        reject(new Error(`Job failed for ${file.name}`));
      } else if (pollCount > 60) { // ~2 min max
        setJobStatusMessage("");
        // Optionally, allow user to continue polling
        reject(Object.assign(new Error(`Job timed out for ${file.name}`), { jobId, file }));
      } else {
        if (message) setJobStatusMessage(message);
        setTimeout(() => pollJobStatus(jobId, file, resolve, reject, pollCount + 1), 2000);
      }
    } catch (err) {
      // Retry a few times on network errors before giving up
      debugLog('Polling error:', err);
      if (networkRetries < 3 && (!err.response || err.code === 'ECONNABORTED' || err.message === 'Network Error')) {
        setTimeout(() => pollJobStatus(jobId, file, resolve, reject, pollCount, networkRetries + 1), 2000);
      } else {
        setJobStatusMessage("");
        reject(err);
      }
    }
  };

  const onDrop = async (acceptedFiles) => {
    debugLog('Files Dropped:', acceptedFiles);
    if (acceptedFiles.length > 1000) {
      setError("❌ You can upload up to 1000 files at a time.");
      return;
    }

    setError("");
    setDownloads([]);
    setCompletedAnalyses(0);
    setTotalFiles(acceptedFiles.length);
    setStatus("uploading");

    let filesRemaining = acceptedFiles.length;

    for (let i = 0; i < acceptedFiles.length; i++) {
      const file = acceptedFiles[i];
      const formData = new FormData();
      formData.append("file", file);
      formData.append("params", JSON.stringify({ custom_red_flags: customRedFlags }));

      try {
        // Upload and get job id
        const res = await apiClient.post("/upload/", formData, {
          onUploadProgress: (progressEvent) => {
            setUploadProgress(Math.round((progressEvent.loaded * 100) / progressEvent.total));
          },
          timeout: 10 * 60 * 1000 // 10 minute timeout for large files
        });
        const job_id = res.data.job_id;
        await new Promise((resolve, reject) => {
          pollJobStatus(job_id, file, resolve, reject);
        })
        .then((result) => {
          // Debug: log result from backend
          console.log('Job result:', result);
          // Always mark as completed, even if some fields are missing
          const isValid =
            result &&
            result.full_analysis_content &&
            result.rep_name &&
            result.categorization;
          if (!isValid) {
            setError(`⚠️ Analysis for ${file.name} is incomplete. Some fields are missing.`);
          }
          // Add to downloads regardless, so user can see partial results
          const filename = result.title || `${result.overall_perf || 'UnknownRating'} ${result.rep_name || 'UnknownRep'} ${result.categorization || 'UnknownType'} ${file.name}.html`;
          const html = buildAnalysisHtml({
            filename,
            full_analysis_content: result.full_analysis_content || '',
            red_flags: result.red_flags || 'No',
            red_flag_reason: result.red_flag_reason || '',
            red_flag_quotes: result.red_flag_quotes || '',
          });
          const blob = new Blob([html], { type: "text/html" });
          const url = URL.createObjectURL(blob);
          setDownloads((prev) => [
            ...prev,
            {
              ...result,
              filename,
              content: html,
              blob,
              url,
            }
          ]);
          setCompletedAnalyses((prev) => prev + 1);
        })
        .catch((err) => {
          debugLog('Polling Error:', err);
          // If job timed out, offer to continue polling
          if (err && err.message && err.message.startsWith('Job timed out')) {
            setError(
              `⏳ Analysis for ${file.name} is taking longer than usual. You can try again or continue polling from the Jobs page.`
            );
          } else if (err.response && err.response.data && err.response.data.result) {
            let details = err.response.data.result.error || err.response.data.result;
            if (err.response.data.result.type) {
              details += `\nType: ${err.response.data.result.type}`;
            }
            setError(`❌ Error processing ${file.name}: ${details}`);
          } else if (err.message) {
            setError(`❌ Error processing ${file.name}: ${err.message}`);
          } else {
            setError(`❌ Error processing ${file.name}: Unknown error`);
          }
        });
      } catch (err) {
        debugLog('Upload Error:', err);
        if (err.code === 'ECONNABORTED' || err.message === 'Network Error') {
          setError(`❌ Upload for ${file.name} failed due to network timeout. Please try again.`);
        } else {
          setError(`❌ Error uploading ${file.name}: ${err.message}`);
        }
      } finally {
        filesRemaining--;
        if (filesRemaining === 0) {
          setStatus("idle");
        }
      }
    }
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    multiple: true,
    maxFiles: 1000,
  });

  // --- PROGRESS BAR & ANALYSIS COUNT FIX ---
  const showUploading = status === "uploading" && uploadProgress < 100;
  const showAnalyzing = completedAnalyses < totalFiles && totalFiles > 0 && !showUploading;

  const handleOpenCoachingForRep = (rep) => {
    setCoachingPrompt("");
    setCoachingResult("");
    setCoachingError("");
    setCoachingRep(rep); 
    console.log('Opening coaching modal for rep:', rep);
    console.log('repStats[rep]:', repStats[rep]);
    const repAnalyses = (repStats[rep] || []).map(dl => {
      if (typeof dl.full_analysis_content === "string" && dl.full_analysis_content.trim()) {
        return dl.full_analysis_content;
      }
      return dl.analysis || dl.transcript || "No analysis available.";
    });
    setCoachingAnalyses(repAnalyses);
    console.log('coachingAnalyses:', repAnalyses);
    setCoachingOpen(true);
  };

  const handleCloseCoaching = () => setCoachingOpen(false);

  const handleGenerateCoaching = async () => {
    const analysesText = coachingAnalyses.length > 0
      ? coachingAnalyses.map((a, i) => `Call Analysis #${i+1}:\n${a}`).join("\n\n")
      : "";
    const fullPrompt = `${coachingPrompt}\n\nHere are this rep's recent call analyses:\n${analysesText}`;
    console.log('Full coaching prompt:', fullPrompt);
    try {
      setCoachingLoading(true);
      setCoachingResult("");
      setCoachingError("");
      const res = await apiClient.post("/api/generate-coaching-plan", { prompt: fullPrompt });
      setCoachingResult(res.data.result);
    } catch (err) {
      setCoachingError(err?.response?.data?.result?.error || err.message || "Unknown error");
    } finally {
      setCoachingLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {!jwt ? (
        <AuthForm onAuth={handleAuth} />
      ) : (
        <Box sx={{ bgcolor: "background.default", minHeight: "100vh" }}>
          <AppBar position="sticky" color="primary" sx={{ mb: 3 }}>
            <Toolbar>
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                {COMPANY} Call Analyzer
              </Typography>
              <Button color="secondary" variant="contained" onClick={handleClearCalls} sx={{ mr: 2 }}>
                Clear Calls
              </Button>
              <Button color="warning" variant="contained" onClick={handleStopAnalysing} disabled={stopAnalysing}>
                Stop Analysing
              </Button>
              <Button color="inherit" startIcon={<LogoutIcon />} onClick={handleLogout} sx={{ ml: 2 }}>
                Logout
              </Button>
            </Toolbar>
          </AppBar>
          <Container maxWidth="lg">
            {/* Rep Stats Section */}
            <Box sx={{ mb: 4 }}>
              <Typography variant="h5" sx={{ mb: 2 }}>
                📊 Rep Stats
              </Typography>
              <Grid container spacing={2}>
                {Object.entries(repStats).map(([rep, calls]) => {
                  const avgScore = (
                    calls.reduce((sum, c) => sum + (parseInt((c.overall_perf || "").split("/")[0], 10) || 0), 0) / calls.length
                  ).toFixed(1);
                  const redFlagCount = calls.filter(c => c.red_flags && c.red_flags.toLowerCase() === "yes").length;
                  // --- NEW: Calculate averages for talk_to_listen_ratio and nos_before_accept ---
                  const avgTalkListen = avgField(calls, "talk_to_listen_ratio");
                  const avgNosBeforeAccept = avgField(calls, "nos_before_accept");
                  return (
                    <Grid item xs={12} sm={6} md={4} key={rep}>
                      <Card elevation={3} sx={{ borderRadius: 3 }}>
                        <CardContent>
                          <Typography variant="h6" sx={{ mb: 1 }}>{rep}</Typography>
                          <Chip label={`Calls: ${calls.length}`} color="info" sx={{ mr: 1 }} />
                          <Chip label={`Avg Score: ${avgScore}`} color="success" sx={{ mr: 1 }} />
                          <Chip label={`Red Flags: ${redFlagCount}`} color={redFlagCount > 0 ? "error" : "success"} sx={{ mr: 1 }} />
                          {/* NEW: Show talk/listen ratio and nos before accept */}
                          <Chip label={`Avg Talk:Listen ${avgTalkListen !== null ? avgTalkListen.toFixed(2) : 'N/A'}`} color="primary" sx={{ mr: 1 }} />
                          <Chip label={`Avg No's Before Accept: ${avgNosBeforeAccept !== null ? avgNosBeforeAccept.toFixed(2) : 'N/A'}`} color="secondary" />
                          <Button variant="outlined" size="small" onClick={() => handleOpenCoachingForRep(rep)} disabled={!(repStats[rep] && repStats[rep].length > 0)} sx={{ mt: 2 }}>
                            Generate Coaching Plan
                          </Button>
                        </CardContent>
                      </Card>
                    </Grid>
                  );
                })}
              </Grid>
            </Box>
            <RedFlagsManager
              jwt={jwt}
              customRedFlags={customRedFlags}
              setCustomRedFlags={setCustomRedFlags}
            />
            <CoachingPlanModal
              open={coachingOpen}
              prompt={coachingPrompt}
              setPrompt={setCoachingPrompt}
              onClose={handleCloseCoaching}
              onGenerate={handleGenerateCoaching}
              result={coachingResult}
              loading={coachingLoading}
              error={coachingError}
            />
            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Upload Call Recordings
                </Typography>
                <Box
                  {...getRootProps()}
                  sx={{
                    border: "2px dashed #00D1FF",
                    borderRadius: 4,
                    p: 4,
                    textAlign: "center",
                    bgcolor: "#F4F6F8",
                    cursor: "pointer",
                    mb: 2
                  }}
                >
                  <input {...getInputProps()} />
                  <CloudUploadIcon sx={{ fontSize: 48, color: "#00D1FF" }} />
                  <Typography variant="body1" sx={{ mt: 2 }}>
                    Drag & drop audio files here, or click to select files
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Supported: mp3, wav, m4a, ogg, etc. (max 1000 files at once)
                  </Typography>
                </Box>
                {showUploading && (
                  <Box sx={{ width: "100%", mt: 1 }}>
                    <LinearProgress variant="determinate" value={uploadProgress} />
                    <Typography variant="body2" align="center" sx={{ mt: 1 }}>
                      {`Uploading... (${uploadProgress}%)`}
                    </Typography>
                    {jobStatusMessage && (
                      <Typography variant="body2" align="center" color="info.main" sx={{ mt: 1 }}>
                        {jobStatusMessage}
                      </Typography>
                    )}
                  </Box>
                )}
                {showAnalyzing && (
                  <Box sx={{ width: "100%", mt: 1 }}>
                    <LinearProgress variant="indeterminate" />
                    <Typography variant="body2" align="center" sx={{ mt: 1 }}>
                      {`Analyzing... (${completedAnalyses}/${totalFiles})`}
                    </Typography>
                    {jobStatusMessage && (
                      <Typography variant="body2" align="center" color="info.main" sx={{ mt: 1 }}>
                        {jobStatusMessage}
                      </Typography>
                    )}
                  </Box>
                )}
                {error && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                  </Alert>
                )}
                {downloads.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Download Reports
                    </Typography>
                    {Object.entries(groupByRep(downloads)).map(([rep, repDownloads]) => (
                      <Box key={rep} sx={{ mb: 3 }}>
                        <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600 }}>
                          {rep}
                        </Typography>
                        <Grid container spacing={2}>
                          {repDownloads.map((dl, idx) => (
                            <Grid item xs={12} sm={6} md={3} key={idx}>
                              <Button
                                fullWidth
                                variant="contained"
                                href={dl.url}
                                download={dl.filename}
                                color={dl.color || "primary"}
                                sx={{
                                  minHeight: 60,
                                  fontWeight: 600,
                                  whiteSpace: "normal",
                                  textTransform: "none"
                                }}
                              >
                                {dl.filename}
                              </Button>
                            </Grid>
                          ))}
                        </Grid>
                      </Box>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Container>
        </Box>
      )}
    </ThemeProvider>
  );
}