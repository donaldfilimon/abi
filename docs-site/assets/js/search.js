// ABI Framework Documentation - Search and Navigation JavaScript

(function() {
    'use strict';

    // DOM Elements
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('sidebar');

    // Documentation index (all pages)
    const docs = [
        {
            title: 'Quick Start',
            url: 'pages/quickstart.html',
            description: 'Get started with ABI in minutes',
            category: 'Getting Started'
        },
        {
            title: 'Introduction',
            url: 'pages/intro.html',
            description: 'Architecture overview and design philosophy',
            category: 'Core'
        },
        {
            title: 'Agent Guide',
            url: 'pages/agents.html',
            description: 'AI agent development patterns and conventions',
            category: 'Developer'
        },
        {
            title: 'API Reference',
            url: 'pages/api.html',
            description: 'Complete API documentation',
            category: 'Reference'
        },
        {
            title: 'AI & Agents',
            url: 'pages/ai.html',
            description: 'LLM connectors and agent runtime',
            category: 'Features'
        },
        {
            title: 'Compute Engine',
            url: 'pages/compute.html',
            description: 'Work-stealing scheduler and task execution',
            category: 'Features'
        },
        {
            title: 'GPU Acceleration',
            url: 'pages/gpu.html',
            description: 'Multi-backend GPU support and unified API',
            category: 'Features'
        },
        {
            title: 'Database',
            url: 'pages/database.html',
            description: 'WDBX vector database',
            category: 'Features'
        },
        {
            title: 'Network',
            url: 'pages/network.html',
            description: 'Distributed compute and Raft consensus',
            category: 'Features'
        },
        {
            title: 'Monitoring',
            url: 'pages/monitoring.html',
            description: 'Logging, metrics, tracing, and profiling',
            category: 'Features'
        },
        {
            title: 'Framework',
            url: 'pages/framework.html',
            description: 'Configuration and lifecycle management',
            category: 'Core'
        },
        {
            title: 'Troubleshooting',
            url: 'pages/troubleshooting.html',
            description: 'Common issues and solutions',
            category: 'Support'
        },
        {
            title: 'Zig 0.16 Migration',
            url: 'pages/migration.html',
            description: 'API changes and compatibility notes',
            category: 'Migration'
        },
        {
            title: 'CLI Commands',
            url: 'pages/cli.html',
            description: 'Complete CLI reference',
            category: 'Reference'
        },
        {
            title: 'Build & Test',
            url: 'pages/build-test.html',
            description: 'Build configuration and testing guide',
            category: 'Core'
        }
    ];

    // Initialize
    function init() {
        setupSearch();
        setupMobileMenu();
        setupActiveLink();
    }

    // Search functionality
    function setupSearch() {
        if (!searchInput) return;

        searchInput.addEventListener('input', debounce(handleSearch, 300));
        searchInput.addEventListener('focus', () => {
            if (searchInput.value.trim()) {
                searchResults.classList.add('show');
            }
        });
        searchInput.addEventListener('blur', () => {
            setTimeout(() => searchResults.classList.remove('show'), 200);
        });
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                searchResults.classList.remove('show');
                searchInput.blur();
            }
        });
    }

    function handleSearch() {
        const query = searchInput.value.trim().toLowerCase();
        if (!query) {
            searchResults.classList.remove('show');
            return;
        }

        const results = searchDocs(query);
        displayResults(results);
        searchResults.classList.add('show');
    }

    function searchDocs(query) {
        const terms = query.split(/\s+/);
        return docs.filter(doc => {
            const title = doc.title.toLowerCase();
            const description = doc.description.toLowerCase();
            const category = doc.category.toLowerCase();
            return terms.every(term =>
                title.includes(term) ||
                description.includes(term) ||
                category.includes(term)
            );
        });
    }

    function displayResults(results) {
        if (results.length === 0) {
            searchResults.innerHTML = `
                <div class="search-result-item">
                    <div class="search-result-title">No results found</div>
                    <div class="search-result-path">Try different search terms</div>
                </div>
            `;
            return;
        }

        const html = results.map(result => `
            <a href="${result.url}" class="search-result-item">
                <div class="search-result-title">${escapeHtml(result.title)}</div>
                <div class="search-result-path">
                    ${escapeHtml(result.category)} • ${escapeHtml(result.description)}
                </div>
            </a>
        `).join('');

        searchResults.innerHTML = html;
    }

    // Mobile menu toggle
    function setupMobileMenu() {
        if (!menuToggle || !sidebar) return;

        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
            menuToggle.textContent = sidebar.classList.contains('open') ? '✕' : '☰';
        });

        // Close sidebar when clicking outside
        document.addEventListener('click', (e) => {
            if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
                sidebar.classList.remove('open');
                menuToggle.textContent = '☰';
            }
        });

        // Close sidebar on window resize
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768) {
                sidebar.classList.remove('open');
                menuToggle.textContent = '☰';
            }
        });
    }

    // Highlight active link in sidebar
    function setupActiveLink() {
        const currentPath = window.location.pathname;
        const sidebarLinks = document.querySelectorAll('.sidebar-link');

        sidebarLinks.forEach(link => {
            if (link.getAttribute('href') === currentPath ||
                (currentPath.includes(link.getAttribute('href')) &&
                 link.getAttribute('href') !== '/')) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }

    // Utility functions
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            searchInput.focus();
        }

        // Escape to close search
        if (e.key === 'Escape') {
            searchResults.classList.remove('show');
            searchInput.blur();
        }
    });

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
