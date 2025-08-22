// EvalNow Analytics - Client-side user tracking
class EvalNowAnalytics {
    constructor() {
        this.userId = this.getOrCreateUserId();
        this.init();
    }

    getOrCreateUserId() {
        let userId = localStorage.getItem('evalnow_user_id');
        if (!userId) {
            userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('evalnow_user_id', userId);
        }
        return userId;
    }

    init() {
        // Track page visit on load
        this.trackPageVisit();
        
        // Track form submissions
        this.trackFormSubmissions();
        
        // Track PDF downloads
        this.trackPdfDownloads();
    }

    async trackEvent(eventType, data = {}) {
        try {
            const eventData = {
                event_type: eventType,
                user_id: this.userId,
                timestamp: new Date().toISOString(),
                url: window.location.href,
                ...data
            };

            await fetch('/api/analytics/track', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(eventData)
            });
        } catch (error) {
            console.log('Analytics tracking failed:', error);
        }
    }

    trackPageVisit() {
        this.trackEvent('page_visit', {
            page: window.location.pathname,
            referrer: document.referrer
        });
    }

    trackFormSubmissions() {
        // Track file upload form
        const uploadForm = document.querySelector('form[action="/evaluation"]');
        if (uploadForm) {
            uploadForm.addEventListener('submit', (e) => {
                const fileInput = uploadForm.querySelector('input[type="file"]');
                if (fileInput && fileInput.files[0]) {
                    this.trackEvent('dataset_upload_started', {
                        filename: fileInput.files[0].name,
                        filesize: fileInput.files[0].size,
                        filetype: fileInput.files[0].type
                    });
                }
            });
        }
    }

    trackPdfDownloads() {
        // Track PDF download links
        document.addEventListener('click', (e) => {
            if (e.target.matches('a[href*="/download-pdf/"]')) {
                this.trackEvent('pdf_download_clicked', {
                    result_id: e.target.href.split('/').pop()
                });
                
                // Add user_id as query parameter to the download URL
                e.preventDefault();
                const originalUrl = e.target.href;
                const separator = originalUrl.includes('?') ? '&' : '?';
                const urlWithUserId = `${originalUrl}${separator}user_id=${encodeURIComponent(this.userId)}`;
                
                // Open the URL with user_id
                window.open(urlWithUserId, '_blank');
            }
        });
    }

    // Track successful evaluation completion
    trackEvaluationCompleted(data) {
        this.trackEvent('dataset_uploaded', {
            dataset_size: data.total_questions,
            filename: data.filename,
            average_score: data.average_score,
            pass_rate: data.pass_rate,
            total_tokens: data.total_tokens || 0,
            input_tokens: data.input_tokens || 0,
            output_tokens: data.output_tokens || 0
        });
        
        // Update local storage stats
        this.updateLocalStats('evaluations');
    }

    // Track PDF download completion
    trackPdfDownloaded(resultId) {
        this.trackEvent('pdf_downloaded', {
            result_id: resultId
        });
        
        this.updateLocalStats('pdf_downloads');
    }

    updateLocalStats(statType) {
        const key = `evalnow_${statType}_count`;
        const current = parseInt(localStorage.getItem(key) || '0');
        localStorage.setItem(key, (current + 1).toString());
    }

    // Get user session stats
    getSessionStats() {
        return {
            userId: this.userId,
            evaluationsCount: parseInt(localStorage.getItem('evalnow_evaluations_count') || '0'),
            pdfDownloadsCount: parseInt(localStorage.getItem('evalnow_pdf_downloads_count') || '0'),
            firstVisit: localStorage.getItem('evalnow_first_visit') || new Date().toISOString()
        };
    }

    // Track first visit timestamp
    trackFirstVisit() {
        if (!localStorage.getItem('evalnow_first_visit')) {
            localStorage.setItem('evalnow_first_visit', new Date().toISOString());
        }
    }
}

// Initialize analytics when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    window.evalNowAnalytics = new EvalNowAnalytics();
    window.evalNowAnalytics.trackFirstVisit();
});

// Expose global function for server-side integration
window.trackEvaluationCompleted = function(data) {
    if (window.evalNowAnalytics) {
        window.evalNowAnalytics.trackEvaluationCompleted(data);
    }
};

window.trackPdfDownloaded = function(resultId) {
    if (window.evalNowAnalytics) {
        window.evalNowAnalytics.trackPdfDownloaded(resultId);
    }
};