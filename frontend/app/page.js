// REVERTED TO BACKUP VERSION WITH GLOBAL DRAG-AND-DROP FOR UPLOADS
"use client";
import React, { useState, useEffect, useRef } from "react";
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
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  Collapse,
  Tooltip,
  CircularProgress
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import LogoutIcon from "@mui/icons-material/Logout";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from '@mui/icons-material/Remove';
import CoachingPlanModal from "./CoachingPlanModal";
import FlagIcon from '@mui/icons-material/Flag';
import PeopleAltIcon from '@mui/icons-material/PeopleAlt';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import ChevronDownIcon from '@mui/icons-material/ExpandMore';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import FlagRoundedIcon from '@mui/icons-material/FlagRounded';
import PeopleAltRoundedIcon from '@mui/icons-material/PeopleAltRounded';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import AssignmentLateIcon from '@mui/icons-material/AssignmentLate';
import WbSunnyIcon from '@mui/icons-material/WbSunny';
import NightlightIcon from '@mui/icons-material/Nightlight';
import { keyframes } from '@mui/system';

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

// Helper to determine button color based on call score
function getCallScoreColor(overall_perf) {
  const score = parseInt((overall_perf || "").split("/")[0], 10);
  if (isNaN(score)) return "primary";
  if (score >= 7) return "success";
  if (score >= 4) return "warning";
  return "error";
}

// Helper to group downloads by normalized representative name, excluding voicemails
function groupByRep(downloads) {
  return downloads.reduce((acc, dl) => {
    // Exclude calls where categorization is 'Voicemail' (case-insensitive, ignore whitespace)
    const cat = (dl.categorization || '').toLowerCase().replace(/\s+/g, '');
    if (cat === 'voicemail') return acc;
    const rep = (dl.rep_name || "Unknown Rep").trim().toLowerCase();
    if (!acc[rep]) acc[rep] = [];
    acc[rep].push(dl);
    return acc;
  }, {});
}

// Helper to average a field
function avgField(arr, field) {
  const vals = arr.map(x => typeof x[field] === 'number' ? x[field] : parseFloat(x[field])).filter(v => !isNaN(v));
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

// Helper to get color by score
function getPerfColor(overall_perf) {
  const score = parseInt((overall_perf || '').split('/')[0], 10);
  if (score >= 7) return 'success';
  if (score >= 4) return 'warning';
  if (score >= 1) return 'error';
  return 'primary';
}

// Helper: Deduplicate downloads by jobId before rendering
function getUniqueDownloads(downloads) {
  const seen = new Set();
  return downloads.filter(dl => {
    if (seen.has(dl.jobId)) return false;
    seen.add(dl.jobId);
    return true;
  });
}

// Helper: When adding to downloads, deduplicate by jobId in state
function addDownloadSafe(downloads, newDl) {
  return downloads.filter(dl => dl.jobId !== newDl.jobId).concat([newDl]);
}

// Helper to extract only the Coaching Plan section from a full analysis string
function extractCoachingPlan(analysis) {
  if (!analysis) return '';
  const match = analysis.match(/Coaching Plan:\s*([\s\S]*?)(?:\n\s*Red Flags:|$)/i);
  return match ? match[1].trim() : '';
}

// ---- ANALYSIS HTML TEMPLATE ----
function buildAnalysisHtml({ filename, full_analysis_content, red_flags, red_flag_reason, red_flag_quotes, missed_custom_red_flags }) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Analysis Report: ${filename}</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; background-color: #F4F6F8; color: #21262E; padding: 20px; }
    .container { max-width: 900px; margin: auto; padding: 25px; background-color: #FFFFFF; box-shadow: 0 4px 10px 0 rgba(0,0,0,0.1); border-radius: 8px; }
    h1, h2 { color: #00D1FF; }
    .analysis-content { line-height: 1.6; font-size: 16px; margin-bottom: 20px; }
    .red-flags { background-color: #FFEBE9; border-left: 5px solid #FF3B30; padding: 15px; border-radius: 5px; margin-top: 20px; }
    blockquote { font-style: italic; background-color: #E9EEF2; padding: 10px; border-radius: 5px; }
    .transcript-block { background: #E9EEF2; padding: 16px; border-radius: 6px; white-space: pre-wrap; font-family: monospace; margin-top: 2em; }
    .missed-flags { background-color: #FFF9E5; border-left: 5px solid #FFB42A; padding: 15px; border-radius: 5px; margin-top: 20px; }
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
    ${
      missed_custom_red_flags && missed_custom_red_flags.length > 0
        ? `<div class="missed-flags">
      <h2>⚠️ Missed Custom Red Flags</h2>
      <ul>${missed_custom_red_flags.map(f => `<li>${f}</li>`).join('')}</ul>
      <p style="color:#B26A00;">These red flags were present in the transcript but not flagged by the AI. Please review them manually.</p>
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
    file: undefined, // Remove any original file reference
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
      // Only use HTML content, never binary
      const html = typeof dl.content === 'string' ? dl.content : '';
      const blob = new Blob([html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      return {
        ...dl,
        blob,
        url,
        full_analysis_content: html,
      };
    });
  } catch {
    return [];
  }
}

// ---- PERSIST JOBS IN LOCAL STORAGE ----
function saveAnalyzingJobsToStorage(jobs) {
  localStorage.setItem('analyzingJobs', JSON.stringify(jobs));
}

function loadAnalyzingJobsFromStorage() {
  try {
    const raw = localStorage.getItem('analyzingJobs');
    if (!raw) return [];
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

function removeAnalyzingJobFromStorage(jobId) {
  const jobs = loadAnalyzingJobsFromStorage();
  const updated = jobs.filter(j => j.jobId !== jobId);
  saveAnalyzingJobsToStorage(updated);
}

// ---- PERSIST ERRORED JOBS IN LOCAL STORAGE ----
function saveErroredJobsToStorage(jobs) {
  localStorage.setItem('erroredJobs', JSON.stringify(jobs));
}

function loadErroredJobsFromStorage() {
  try {
    const raw = localStorage.getItem('erroredJobs');
    if (!raw) return [];
    return JSON.parse(raw);
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
  const [input, setInput] = useState("");
  const [customRedFlags, setCustomRedFlags] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('customRedFlags') || '[]');
    } catch { return []; }
  });
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
  const [analyzingJobs, setAnalyzingJobs] = useState(() => loadAnalyzingJobsFromStorage());
  const [erroredJobs, setErroredJobs] = useState(() => loadErroredJobsFromStorage());
  const [redFlagsOpen, setRedFlagsOpen] = useState(false);
  const [repsOpen, setRepsOpen] = useState(false);
  const [repInput, setRepInput] = useState("");
  const [currentReps, setCurrentReps] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('currentReps') || '[]');
    } catch { return []; }
  });
  const [dragActive, setDragActive] = useState(false);
  const [toast, setToast] = useState({ open: false, message: '', severity: 'success' });
  const [uploadError, setUploadError] = useState('');
  const [userInfo, setUserInfo] = useState({ email: '' });

  // --- Robust red flag stats state ---
  const [robustRedFlagStats, setRobustRedFlagStats] = useState({});

  useEffect(() => {
    localStorage.setItem('customRedFlags', JSON.stringify(customRedFlags));
  }, [customRedFlags]);
  useEffect(() => {
    localStorage.setItem('currentReps', JSON.stringify(currentReps));
  }, [currentReps]);

  useEffect(() => {
    saveDownloadsToStorage(downloads);
  }, [downloads]);

  useEffect(() => {
    saveAnalyzingJobsToStorage(analyzingJobs);
  }, [analyzingJobs]);

  useEffect(() => {
    saveErroredJobsToStorage(erroredJobs);
  }, [erroredJobs]);

  useEffect(() => {
    setDownloads(loadDownloadsFromStorage());
    setAnalyzingJobs(loadAnalyzingJobsFromStorage());
    setErroredJobs(loadErroredJobsFromStorage());
  }, []);

  // --- Fetch robust red flag stats for each rep ---
  // useEffect(() => {
  //   if (!jwt) return; // Only run if logged in
  //   if (Object.keys(repStats).length === 0) return;
  //   (async () => {
  //     for (const [rep, calls] of Object.entries(repStats)) {
  //       // Defensive: filter out undefined/null/malformed calls
  //       const filteredCalls = calls.filter(call => call && typeof call === 'object');
  //       try {
  //         console.log('Posting job_results:', filteredCalls); // Debug log
  //         const res = await apiClient.post('/api/rep-red-flag-stats', { job_results: filteredCalls });
  //         console.log('Received stats:', res.data); // Debug log
  //         setRobustRedFlagStats(prev => ({ ...prev, [rep]: res.data }));
  //       } catch (err) {
  //         setRobustRedFlagStats(prev => ({ ...prev, [rep]: { total_red_flags: 0 } }));
  //       }
  //     }
  //   })();
  // }, [jwt, repStats]);

  // ---- GLOBAL ACTIVE POLL TRACKER ----
  const activeJobPolls = {};

  // Utility to clear all active job polls
  function clearAllActiveJobPolls() {
    Object.keys(activeJobPolls).forEach(jobId => delete activeJobPolls[jobId]);
  }

  // --- Robust polling for analyzing jobs ---
  const POLL_INTERVAL_MS = 5000;  // 5s is plenty for GPT

  function pollJobUntilDone(job) {
    if (activeJobPolls[job.jobId]) return; // Prevent duplicate polling
    activeJobPolls[job.jobId] = true;
    const poll = async () => {
      try {
        const res = await apiClient.get(`/job-status/${job.jobId}`);
        const { status: jobStatus, result } = res.data;
        setJobProgress(prev => ({ ...prev, [job.jobId]: { status: jobStatus, result } }));
        if (jobStatus === "done") {
          const fileName = result?.fileName || job.fileName || "Analysis Report.html";
          const html = buildAnalysisHtml({
            filename: fileName,
            full_analysis_content: result?.full_analysis_content || '',
            red_flags: result?.red_flags,
            red_flag_reason: result?.red_flag_reason,
            red_flag_quotes: result?.red_flag_quotes,
            missed_custom_red_flags: result?.missed_custom_red_flags || []
          });
          const blob = new Blob([html], { type: 'text/html' });
          const url = URL.createObjectURL(blob);
          const debug_custom = Array.isArray(result?.custom_red_flags_found) ? result.custom_red_flags_found.length : 0;
          const debug_system = Array.isArray(result?.system_red_flags_found) ? result.system_red_flags_found.length : 0;
          console.log('JOB COMPLETE:', job.jobId, 'Custom Flags:', debug_custom, result?.custom_red_flags_found, 'System Flags:', debug_system, result?.system_red_flags_found);
          setDownloads(prev => addDownloadSafe(prev, {
            ...result,
            jobId: job.jobId,
            fileName,
            blob,
            url,
            full_analysis_content: result?.full_analysis_content || '',
            displayName: fileName,
            // Ensure custom_red_flags_found and system_red_flags_found are properly set
            custom_red_flags_found: result?.custom_red_flags_found || [],
            system_red_flags_found: result?.system_red_flags_found || [],
          }));
          removeAnalyzingJobFromStorage(job.jobId);
          setAnalyzingJobs(loadAnalyzingJobsFromStorage());
          delete activeJobPolls[job.jobId];
        } else if (jobStatus === "error") {
          // Move to errored jobs
          const errored = loadErroredJobsFromStorage();
          const erroredJob = { ...job, result };
          saveErroredJobsToStorage([...errored, erroredJob]);
          setErroredJobs(loadErroredJobsFromStorage());
          delete activeJobPolls[job.jobId];
        } else {
          // Not done/error: schedule next poll
          if (activeJobPolls[job.jobId]) {
            setTimeout(poll, POLL_INTERVAL_MS);
          }
        }
      } catch (err) {
        setJobProgress(prev => ({ ...prev, [job.jobId]: { status: "error", error: err } }));
        delete activeJobPolls[job.jobId];
      }
    };
    poll();
  }

  // --- Start polling for all analyzingJobs not already completed ---
  useEffect(() => {
    if (!jwt) return;
    analyzingJobs.forEach(job => {
      if (!downloads.some(dl => dl.jobId === job.jobId)) {
        pollJobUntilDone(job);
      }
    });
  }, [analyzingJobs, jwt]);

  // ---- ENV VARS ----
  const COMPANY = process.env.NEXT_PUBLIC_COMPANY || "Greener Living";

  // --- Rep stats calculation ---
  const repStats = groupByRep(getUniqueDownloads(downloads));

  // --- Clear Calls ---
  const handleClearCalls = () => {
    setDownloads([]);
    localStorage.setItem('downloads', JSON.stringify([]));
  };

  const handleClearFailedAnalyses = () => {
    setErroredJobs([]);
    localStorage.setItem('erroredJobs', JSON.stringify([]));
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
    // Clear job-related localStorage on new login
    localStorage.removeItem('analyzingJobs');
    localStorage.removeItem('downloads');
    localStorage.removeItem('erroredJobs');
    // Clear all in-memory job polling
    clearAllActiveJobPolls();
    // Add any other job/user-specific keys if needed
    localStorage.setItem("jwt", token);
    setJwt(token);
  };

  const handleLogout = () => {
    debugLog('Logout');
    localStorage.removeItem("jwt");
    setJwt(null);
    setMinutes(null);
    setDownloads([]);
    // Clear all in-memory job polling
    clearAllActiveJobPolls();
  };

  useEffect(() => {
    if (jwt) {
      apiClient.get("/users/me").then(res => {
        setMinutes(res.data.minutes);
        setUserInfo({ email: res.data.email });
        localStorage.setItem('userEmail', res.data.email || '');
      });
    }
  }, [jwt, completedAnalyses]);

  useEffect(() => {
    saveDownloadsToStorage(downloads);
  }, [downloads]);

  const onDrop = async (acceptedFiles) => {
    if (!acceptedFiles.length) {
      setToast({ open: true, message: 'No files selected.', severity: 'warning' });
      return;
    }
    if (acceptedFiles.length > 1000) {
      setUploadError('❌ You can upload up to 1000 files at a time.');
      setToast({ open: true, message: 'Too many files selected (max 1000).', severity: 'error' });
      return;
    }
    // File type/size validation
    const invalid = acceptedFiles.find(f => f.size > 25 * 1024 * 1024);
    if (invalid) {
      setUploadError('File too large (max 25MB): ' + invalid.name);
      setToast({ open: true, message: 'File too large: ' + invalid.name, severity: 'error' });
      return;
    }
    setTotalFiles(acceptedFiles.length);
    setStatus('uploading');
    setToast({ open: true, message: 'Upload started!', severity: 'info' });
    setUploadProgress(0);
    setCompletedAnalyses(0);
    setUploadError('');

    const formData = new FormData();
    acceptedFiles.forEach(f => formData.append('files', f));
    formData.append('params', JSON.stringify({
      custom_red_flags: customRedFlags,
      current_reps: currentReps
    }));

    try {
      const res = await apiClient.post('/upload/', formData, {
        timeout: 10 * 60 * 1000,
        onUploadProgress: e => {
          if (e.total) setUploadProgress(Math.round((e.loaded * 100) / e.total));
        }
      });
      const { job_ids } = res.data;
      job_ids.forEach((jobId, idx) => {
        const job = { jobId, fileName: acceptedFiles[idx].name };
        const jobs = loadAnalyzingJobsFromStorage();
        saveAnalyzingJobsToStorage([...jobs, job]);
        setAnalyzingJobs(loadAnalyzingJobsFromStorage());
        pollJobUntilDone(job);
      });
    } catch (err) {
      if (err.response?.status === 413) {
        setUploadError('❌ Upload too large.');
      } else if (err.response?.status === 401) {
        setUploadError('❌ Session expired. Please log in again.');
        localStorage.removeItem('jwt');
        setJwt(null);
      } else if (err.message && err.message.includes('Network Error')) {
        setUploadError('❌ Network error. Please check your connection or try again later.');
      } else {
        setUploadError(`❌ Upload failed: ${err.response?.data?.detail || err.message}`);
      }
      setToast({ open: true, message: 'Upload failed.', severity: 'error' });
      setStatus('idle');
      return;
    }

    setStatus('idle');
    setToast({ open: true, message: 'Upload complete!', severity: 'success' });
    setUploadProgress(0);
    setCompletedAnalyses(0);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "audio/*": [".mp3", ".wav", ".m4a", ".aac", ".flac"]
    },
    multiple: true
  });

  const pulse = keyframes`
    0% { transform: scale(1); filter: drop-shadow(0 0 0 #00d1ff44); }
    50% { transform: scale(1.08); filter: drop-shadow(0 0 10px #00d1ff); }
    100% { transform: scale(1); filter: drop-shadow(0 0 0 #00d1ff44); }
  `;

  const uploadZoneRef = useRef(null);

  // --- PROGRESS BAR & ANALYSIS COUNT FIX ---
  const showUploading = status === "uploading" && uploadProgress > 0;
  const showAnalyzing = completedAnalyses < totalFiles && totalFiles > 0 && !showUploading;

  const handleOpenCoachingForRep = (rep) => {
    setCoachingPrompt("");
    setCoachingResult("");
    setCoachingError("");
    setCoachingRep(rep); 
    console.log('Opening coaching modal for rep:', rep);
    console.log('repStats[rep]:', repStats[rep]);
    const repAnalyses = (repStats[rep] || []).map(dl => {
      let analysis = typeof dl.full_analysis_content === "string" && dl.full_analysis_content.trim()
        ? dl.full_analysis_content
        : dl.analysis || dl.transcript || "No analysis available.";
      return extractCoachingPlan(analysis);
    }).filter(Boolean); // Only keep non-empty coaching plans
    setCoachingAnalyses(repAnalyses);
    console.log('coachingAnalyses:', repAnalyses);
    setCoachingOpen(true);
  };

  const handleCloseCoaching = () => setCoachingOpen(false);

  const handleGenerateCoaching = async () => {
    // Only use the coaching plans, not full analyses
    const analysesText = coachingAnalyses.length > 0
      ? coachingAnalyses.map((a, i) => `Coaching Plan #${i+1}:\n${a}`).join("\n\n")
      : "";
    const fullPrompt = `${coachingPrompt}\n\nHere are this rep's recent coaching plans:\n${analysesText}`;
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

  // --- Side Nav & Modal State ---
  const handleOpenRedFlags = () => setRedFlagsOpen(true);
  const handleOpenReps = () => setRepsOpen(true);

  const showToast = (message, severity = 'success') => setToast({ open: true, message, severity });

  const userEmail = (typeof window !== 'undefined' && localStorage.getItem('userEmail')) || '';

  // --- Download handler to always use full HTML and robust filename extension ---
  function handleDownloadReport(dl) {
    // Revoke previous URL if it exists to avoid memory leaks
    if (dl.url) {
      URL.revokeObjectURL(dl.url);
    }
    // Always generate a full HTML document using buildAnalysisHtml if needed
    let html;
    if (dl.full_analysis_content && dl.full_analysis_content.startsWith('<!DOCTYPE html>')) {
      html = dl.full_analysis_content;
    } else if (typeof buildAnalysisHtml === 'function') {
      html = buildAnalysisHtml({
        filename: dl.fileName || 'Analysis Report',
        full_analysis_content: dl.full_analysis_content || dl.content || '',
        red_flags: dl.red_flags,
        red_flag_reason: dl.system_red_flag_reason,
        red_flag_quotes: dl.system_red_flag_quotes,
        missed_custom_red_flags: dl.missed_custom_red_flags || []
      });
    } else {
      html = dl.full_analysis_content || dl.content || '';
    }
    // Ensure filename ends with .html
    let fileName = dl.fileName || 'Analysis Report.html';
    if (!fileName.endsWith('.html')) fileName += '.html';
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }, 1000);
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {jwt && (
        <Drawer
          variant="permanent"
          sx={{
            width: 240,
            flexShrink: 0,
            whiteSpace: 'nowrap',
            boxSizing: 'border-box',
            bgcolor: "#f8fafc",
            borderRight: '1px solid #e0e4ea',
            '& .MuiDrawer-paper': {
              width: 240,
              bgcolor: "#f8fafc",
              borderRight: '1px solid #e0e4ea',
              transition: theme.transitions.create('width', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.enteringScreen,
              }),
              overflowX: 'hidden',
              boxShadow: '2px 0 8px 0 rgba(30,40,90,0.03)',
            },
          }}
        >
          {/* White space above orange line */}
          <Box sx={{ width: '100%', height: 32, bgcolor: '#fff' }} />
          <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Minutes & Buy Minutes */}
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 1, mb: 2, px: 2, mt: 0 }}>
              <Chip
                label={
                  minutes === null
                    ? 'Loading minutes…'
                    : `${minutes} minute${minutes === 1 ? '' : 's'} left`
                }
                color={minutes > 0 ? 'success' : 'warning'}
                sx={{
                  mr: 2,
                  bgcolor: '#f5f7fa',
                  color: '#2b3a55',
                  border: '1px solid #e0e4ea',
                  fontWeight: 600,
                  fontSize: 14,
                  height: 36,
                  borderRadius: 2,
                  px: 2,
                  mb: 1,
                  mt: 0,
                  '& .MuiChip-label': { fontSize: 14, fontWeight: 600, fontFamily: 'Inter, sans-serif' },
                }}
              />
              <Button
                variant="contained"
                color="success"
                sx={{
                  borderRadius: 2,
                  fontWeight: 600,
                  fontSize: 14,
                  minWidth: 120,
                  bgcolor: '#f5f7fa',
                  color: '#2b3a55',
                  border: '1px solid #e0e4ea',
                  boxShadow: 'none',
                  textTransform: 'none',
                  height: 36,
                  mt: 0,
                  mb: 1,
                  '&:hover': { bgcolor: '#e3e8f0', borderColor: '#b6c3d1', color: '#2b3a55' },
                  fontFamily: 'Inter, sans-serif',
                }}
                onClick={() => setBuyDialogOpen(true)}
              >
                Buy Minutes
              </Button>
            </Box>
            {/* Sidebar navigation sections */}
            <List sx={{px: 1, pt: 2, gap: 2, flexGrow: 1}}>
              {/* Custom Red Flags Section */}
              <Tooltip title="Custom Red Flags" placement="right" arrow disableHoverListener={true}>
                <ListItem disablePadding sx={{ display: 'block', borderRadius: 2, mb: 1, borderLeft: redFlagsOpen ? '4px solid #00D1FF' : '4px solid transparent', transition: 'all 0.2s' }}>
                  <ListItemButton onClick={() => setRedFlagsOpen(!redFlagsOpen)} sx={{ minHeight: 44, borderRadius: 2, px: 2.5, gap: 1.5 }}>
                    <FlagRoundedIcon sx={{ fontSize: 24, color: '#00D1FF' }} />
                    <ListItemText primary="Custom Red Flags" sx={{ ml: 1, color: '#0F2A54', fontWeight: 600 }} />
                  </ListItemButton>
                </ListItem>
              </Tooltip>
              <Collapse in={redFlagsOpen} timeout={300} unmountOnExit>
                <Box sx={{ px: 3, py: 2, bgcolor: '#fff', borderRadius: 2, boxShadow: '0 2px 8px 0 rgba(30,40,90,0.03)' }}>
                  <TextField
                    placeholder="Add Red Flag"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && input.trim()) { setCustomRedFlags([...customRedFlags, input.trim()]); setInput(''); }}}
                    size="small"
                    fullWidth
                    sx={{ mb: 1, bgcolor: '#f5f7fa', borderRadius: 2, '& .MuiOutlinedInput-root': { fontSize: 14, height: 36 } }}
                    InputLabelProps={{ shrink: false }}
                    inputProps={{ style: { borderRadius: 8, padding: '8px 10px' } }}
                  />
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, minHeight: 24 }}>
                    {customRedFlags.map((flag, idx) => (
                      <Chip
                        key={idx}
                        label={flag}
                        onDelete={() => setCustomRedFlags(customRedFlags.filter((_, i) => i !== idx))}
                        sx={{ bgcolor: '#f5f7fa', color: '#2b3a55', border: '1px solid #e0e4ea', fontWeight: 600, fontSize: 13, height: 28, transition: 'all 0.2s', '&:hover': { bgcolor: '#e3e8f0', borderColor: '#b6c3d1' }, boxShadow: '0 1px 3px 0 rgba(30,40,90,0.03)' }}
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              </Collapse>
              {/* Current Reps Section */}
              <Tooltip title="Current Reps" placement="right" arrow disableHoverListener={true}>
                <ListItem disablePadding sx={{ display: 'block', borderRadius: 2, mb: 1, borderLeft: repsOpen ? '4px solid #00D1FF' : '4px solid transparent', transition: 'all 0.2s' }}>
                  <ListItemButton onClick={() => setRepsOpen(!repsOpen)} sx={{ minHeight: 44, borderRadius: 2, px: 2.5, gap: 1.5 }}>
                    <PeopleAltRoundedIcon sx={{ fontSize: 24, color: '#00D1FF' }} />
                    <ListItemText primary="Current Reps" sx={{ ml: 1, color: '#0F2A54', fontWeight: 600 }} />
                  </ListItemButton>
                </ListItem>
              </Tooltip>
              <Collapse in={repsOpen} timeout={300} unmountOnExit>
                <Box sx={{ px: 3, py: 2, bgcolor: '#fff', borderRadius: 2, boxShadow: '0 2px 8px 0 rgba(30,40,90,0.03)' }}>
                  <TextField
                    placeholder="Add Rep Name"
                    value={repInput}
                    onChange={e => setRepInput(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && repInput.trim()) { setCurrentReps([...currentReps, repInput.trim()]); setRepInput(''); }}}
                    size="small"
                    fullWidth
                    sx={{ mb: 1, bgcolor: '#f5f7fa', borderRadius: 2, '& .MuiOutlinedInput-root': { fontSize: 14, height: 36 } }}
                    InputLabelProps={{ shrink: false }}
                    inputProps={{ style: { borderRadius: 8, padding: '8px 10px' } }}
                  />
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, minHeight: 24 }}>
                    {currentReps.map((rep, idx) => (
                      <Chip
                        key={idx}
                        label={rep}
                        onDelete={() => setCurrentReps(currentReps.filter((_, i) => i !== idx))}
                        sx={{ bgcolor: '#f5f7fa', color: '#2b3a55', border: '1px solid #e0e4ea', fontWeight: 600, fontSize: 13, height: 28, transition: 'all 0.2s', '&:hover': { bgcolor: '#e3e8f0', borderColor: '#b6c3d1' }, boxShadow: '0 1px 3px 0 rgba(30,40,90,0.03)' }}
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              </Collapse>
            </List>
            {/* Logout at the bottom */}
            <Box sx={{ flexShrink: 0, mb: 2, px: 2 }}>
              <Divider sx={{ mb: 2 }} />
              <Box sx={{ height: 48 }} />
              {userInfo.email && (
                <Typography sx={{ fontSize: 14, color: '#0f2a54', fontWeight: 600, mb: 1, textAlign: 'left' }}>
                  {userInfo.email}
                </Typography>
              )}
              <Box sx={{ height: 8 }} />
              <Button
                startIcon={<LogoutIcon />}
                onClick={handleLogout}
                fullWidth
                color="error"
                variant="outlined"
                sx={{ borderRadius: 99, fontWeight: 600, fontSize: 13 }}
              >
                Logout
              </Button>
            </Box>
          </Box>
        </Drawer>
      )}
      <Box sx={{ flexGrow: 1, ml: jwt ? 30 : 0 }}>
        {!jwt ? (
          <AuthForm onAuth={handleAuth} />
        ) : (
          <Box sx={{ bgcolor: "background.default", minHeight: "100vh" }}>
            <Container maxWidth="lg">
              <Box sx={{ height: 16 }} />
              {Object.keys(repStats).length > 0 && (
                <Box sx={{ mb: 4 }}>
                  {Object.entries(repStats).map(([rep, calls]) => {
                    console.log("Talk-to-listen ratios for", rep, ":", calls.map(c => c.talk_to_listen_ratio));
                    const avgScore = (
                      calls.reduce((sum, c) => sum + (parseInt((c.overall_perf || "").split("/")[0], 10) || 0), 0) / calls.length
                    ).toFixed(1);
                    // Use robust red flag count
                    const redFlagCount = calls.reduce((sum, c) => {
                      const sys = typeof c.system_red_flags_count === 'number' ? c.system_red_flags_count : (c.system_red_flags_count || 0);
                      const custom = Array.isArray(c.custom_red_flags_found) ? c.custom_red_flags_found.length : 0;
                      return sum + sys + custom;
                    }, 0);
                    const avgTalkListen = avgField(calls, "talk_to_listen_ratio");
                    const avgNosBeforeAccept = avgField(calls, "nos_before_accept");
                    // Count each call type (categorization)
                    const callTypeCounts = calls.reduce((acc, c) => {
                      const type = (c.categorization || 'Unknown').trim();
                      acc[type] = (acc[type] || 0) + 1;
                      return acc;
                    }, {});
                    return (
                      <Grid item xs={12} sm={6} md={4} key={rep}>
                        <Card elevation={3} sx={{ borderRadius: 3 }}>
                          <CardContent>
                            <Typography variant="h6" sx={{ mb: 1 }}>{rep}</Typography>
                            <Chip label={`Calls: ${calls.length}`} color="info" sx={{ mr: 1 }} />
                            <Chip label={`Avg Score: ${avgScore}`} color="success" sx={{ mr: 1 }} />
                            <Chip label={`Red Flags: ${redFlagCount}`} color={redFlagCount > 0 ? "error" : "success"} sx={{ mr: 1 }} />
                            <Chip
                              label={`Rep talks for: ${avgTalkListen !== null ? Math.round(avgTalkListen) + '%' : 'N/A'}`}
                              color="primary"
                              sx={{ mr: 1 }}
                            />
                            <Chip
                              label={`Avg No's Before Accept: ${avgNosBeforeAccept !== null ? avgNosBeforeAccept.toFixed(2) : 'N/A'}`}
                              color="secondary"
                            />
                            <Box sx={{ mt: 1, mb: 1 }}>
                              {Object.entries(callTypeCounts).map(([type, count]) => (
                                <Chip key={type} label={`${type}: ${count}`} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                              ))}
                            </Box>
                            <Button variant="outlined" size="small" onClick={() => handleOpenCoachingForRep(rep)} disabled={!(repStats[rep] && repStats[rep].length > 0)} sx={{ mt: 2 }}>
                              Generate Coaching Plan
                            </Button>
                          </CardContent>
                        </Card>
                      </Grid>
                    );
                  })}
                </Box>
              )}
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
              <Card sx={{ mt: 2, mb: 4, borderRadius: 4, boxShadow: '0 6px 24px 0 rgba(0,209,255,0.07)', p: 0, overflow: 'visible' }}>
                <CardContent sx={{ p: { xs: 2, sm: 4 }, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                  <Typography variant="h4" sx={{ fontFamily: 'Inter, sans-serif', fontWeight: 800, mb: 1, letterSpacing: 0.2 }}>
                    Upload Call Recordings
                  </Typography>
                  <Box
                    {...getRootProps()}
                    aria-label="Upload call recordings"
                    tabIndex={0}
                    sx={{
                      borderRadius: 3,
                      background: 'linear-gradient(135deg, #e6faff 0%, #f6fcff 100%)',
                      boxShadow: '0 2px 12px 0 rgba(0,209,255,0.10)',
                      border: 'none',
                      outline: '2px dashed #00d1ff44',
                      outlineOffset: '-8px',
                      p: { xs: 3, sm: 5 },
                      width: '100%',
                      maxWidth: 700,
                      minHeight: 180,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      position: 'relative',
                      transition: 'box-shadow 0.3s, background 0.3s',
                      '&:hover': {
                        boxShadow: '0 8px 32px 0 #00d1ff33',
                        background: 'linear-gradient(135deg, #e6faff 0%, #e0f7fa 100%)',
                      }
                    }}
                    ref={uploadZoneRef}
                  >
                    <input {...getInputProps()} aria-label="Select files to upload" />
                    <CloudUploadIcon sx={{ fontSize: 64, color: '#00d1ff', mb: 2, animation: `${pulse} 1.5s infinite` }} />
                    <Typography variant="h6" sx={{ fontFamily: 'Inter, sans-serif', fontWeight: 700, mb: 1 }}>
                      Drag & drop audio files here, or click to select files
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
                      <Typography variant="body2" sx={{ color: "#6b7a90", fontFamily: 'Roboto, sans-serif' }}>
                        Supported: mp3, wav, m4a, etc. (max 1000 files at once)
                      </Typography>
                      <Tooltip title="Max file size: 25MB. Supported: mp3, wav, m4a, ogg. Drag anywhere!">
                        <InfoOutlinedIcon sx={{ fontSize: 18, color: '#00d1ff', ml: 0.5, cursor: 'pointer' }} />
                      </Tooltip>
                    </Box>
                    {status === 'uploading' && (
                      <Box sx={{ width: '100%', mt: 2 }}>
                        <LinearProgress variant="determinate" value={uploadProgress} sx={{ height: 8, borderRadius: 4, background: '#e0f7fa', '& .MuiLinearProgress-bar': { background: 'linear-gradient(90deg, #00C36F 0%, #00D1FF 100%)' } }} />
                        <Typography variant="body2" sx={{ mt: 1, color: '#00C36F', fontWeight: 600, fontFamily: 'Roboto, sans-serif' }}>
                          Uploading {completedAnalyses + 1} of {totalFiles} files...
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </CardContent>
              </Card>

              {/* --- In Progress Section: Add Stop Analysing button --- */}
              {analyzingJobs.length > 0 && (
                <Box sx={{ width: '100%', mt: 2 }}>
                  <Typography variant="h6" gutterBottom>In Progress</Typography>
                  <Button
                    variant="outlined"
                    color="error"
                    sx={{ mb: 2 }}
                    onClick={handleStopAnalysing}
                  >
                    Stop Analyzing
                  </Button>
                  {analyzingJobs.map(job => (
                    <Box key={job.jobId} sx={{ mb: 2, p: 2, bgcolor: '#FFF3E0', borderRadius: 2 }}>
                      <Typography variant="body2" fontWeight={600}>{job.fileName}</Typography>
                      <Typography variant="body2" color="text.secondary">Status: Analyzing...</Typography>
                    </Box>
                  ))}
                </Box>
              )}
              {(downloads.length === 0 && analyzingJobs.length === 0 && status !== 'uploading') ? (
                <Box sx={{ mt: 6, mb: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                  <AssignmentLateIcon sx={{ fontSize: 72, color: '#b6c3d1', mb: 1 }} />
                  <Typography variant="h6" sx={{ fontWeight: 700, color: '#7e8ba3', mb: 1 }}>
                    No completed analyses yet
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#7e8ba3', mb: 2, fontFamily: 'Roboto, sans-serif' }}>
                    Upload files to see your first report.
                  </Typography>
                </Box>
              ) : (
                <Box sx={{ mt: 4 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">
                      Download Reports
                    </Typography>
                    {downloads.length > 0 && (
                      <Button
                        variant="outlined"
                        color="error"
                        onClick={handleClearCalls}
                        size="small"
                      >
                        Clear Calls
                      </Button>
                    )}
                  </Box>
                  {Object.keys(groupByRep(getUniqueDownloads(downloads))).length === 0 && (
                    <Typography variant="body2" color="text.secondary">
                      No completed analyses yet.
                    </Typography>
                  )}
                  {downloads.length > 0 && (
                    <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
                    </Box>
                  )}
                  {Object.entries(groupByRep(getUniqueDownloads(downloads))).map(([rep, downloads]) => (
                    <Box key={rep} sx={{ mb: 2 }}>
                      <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>{rep}</Typography>
                      {downloads.map(dl => (
                        <Box key={dl.jobId} sx={{ mb: 1 }}>
                          <Button
                            variant="contained"
                            color={getCallScoreColor(dl.overall_perf)}
                            onClick={() => handleDownloadReport(dl)}
                            sx={{
                              width: '100%',
                              display: 'flex',
                              justifyContent: 'flex-start',
                              textTransform: 'none',
                              fontWeight: 600
                            }}
                          >
                            {dl.displayName || dl.fileName || 'Download Report'}
                          </Button>
                          {dl && (
                            <Box sx={{ mt: 1, mb: 1, pl: 2 }}>
                            <Typography variant="subtitle2" sx={{ color: '#0f2a54', fontWeight: 600 }}>
                              System Red Flags: {Array.isArray(dl.system_red_flags_found) ? dl.system_red_flags_found.length : 0}
                            </Typography>
                            {Array.isArray(dl.system_red_flags_found) && dl.system_red_flags_found.length > 0 && (
                              <Box sx={{ mb: 1, pl: 1 }}>
                                {dl.system_red_flags_found.map((flag, idx) => (
                                  <Typography key={idx} variant="body2" sx={{ color: '#b71c1c' }}>
                                    {flag}
                                  </Typography>
                                ))}
                              </Box>
                            )}
                            <Typography variant="subtitle2" sx={{ color: '#0f2a54', fontWeight: 600 }}>
                              Custom Red Flags: {Array.isArray(dl.custom_red_flags_found) ? dl.custom_red_flags_found.length : 0}
                            </Typography>
                            {Array.isArray(dl.custom_red_flags_found) && dl.custom_red_flags_found.length > 0 && (
                              <div style={{ marginLeft: 12 }}>
                                {dl.custom_red_flags_found.map((flag, idx) => (
                                  <div key={idx} style={{ marginBottom: '4px', color: '#b71c1c', fontSize: '0.95em' }}>
                                    <span style={{ fontWeight: 500 }}>{flag}</span>
                                    {dl.custom_red_flags_explanations && dl.custom_red_flags_explanations[flag] && (
                                      <span style={{ marginLeft: '8px', fontStyle: 'italic' }}>
                                        — {dl.custom_red_flags_explanations[flag]}
                                      </span>
                                    )}
                                  </div>
                                ))}
                              </div>
                            )}
                          </Box>
                          )}
                        </Box>
                      ))}
                    </Box>
                  ))}
                </Box>
              )}
              {erroredJobs.length > 0 && (
                <Box sx={{ mt: 4 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">
                      Failed Analyses
                    </Typography>
                    {erroredJobs.length > 0 && (
                      <Button
                        variant="outlined"
                        color="error"
                        onClick={handleClearFailedAnalyses}
                        size="small"
                      >
                        Clear Failed Analyses
                      </Button>
                    )}
                  </Box>
                  {erroredJobs.map(job => (
                    <Box key={job.jobId} sx={{ mb: 2, p: 2, bgcolor: '#FFF3E0', borderRadius: 2 }}>
                      <Typography variant="body2" fontWeight={600}>{job.fileName}</Typography>
                      <Typography variant="body2" color="text.secondary">Status: Error</Typography>
                      <Typography variant="body2" color="error">{job.result?.error || 'Unknown error'}</Typography>
                    </Box>
                  ))}
                </Box>
              )}
            </Container>
          </Box>
        )}
      </Box>
      <Snackbar open={toast.open} autoHideDuration={4000} onClose={() => setToast({ ...toast, open: false })} anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
        <Alert elevation={6} variant="filled" onClose={() => setToast({ ...toast, open: false })} severity={toast.severity} sx={{ fontWeight: 600 }}>
          {toast.message}
        </Alert>
      </Snackbar>
      {uploadError && (
        <Snackbar open={!!uploadError} autoHideDuration={6000} onClose={() => setUploadError('')} anchorOrigin={{ vertical: 'top', horizontal: 'center' }}>
          <Alert elevation={6} variant="filled" severity="error" onClose={() => setUploadError('')} sx={{ fontWeight: 600 }}>
            {uploadError}
          </Alert>
        </Snackbar>
      )}
      {showAnalyzing && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
          <CircularProgress size={22} sx={{ color: '#00d1ff' }} />
          <Typography variant="body2" sx={{ fontWeight: 500, color: '#00d1ff' }}>Analyzing... ({completedAnalyses}/{totalFiles})</Typography>
        </Box>
      )}
      {showUploading && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
          <CircularProgress size={22} sx={{ color: '#00d1ff' }} />
          <Typography variant="body2" sx={{ fontWeight: 500, color: '#00d1ff' }}>Uploading... ({uploadProgress}%)</Typography>
        </Box>
      )}
      {analyzingJobs.length > 0 && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
          <Button
            variant="outlined"
            color="error"
            sx={{ mt: 2 }}
            onClick={handleStopAnalysing}
          >
            Stop Analyzing
          </Button>
          <Typography variant="body2" sx={{ fontWeight: 500, color: '#00d1ff' }}>{completedAnalyses}/{totalFiles} files analyzed.</Typography>
        </Box>
      )}
      {downloads.length > 0 && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
          <Button
            variant="outlined"
            color="error"
            onClick={handleClearCalls}
            sx={{ mt: 2 }}
          >
            Clear Calls
          </Button>
        </Box>
      )}
    </ThemeProvider>
  );
}