/**
 * ABI Documentation - Main JavaScript
 * Handles theme toggling, navigation, and interactive features
 */

(function() {
  'use strict';

  // ==========================================================================
  // Theme Management
  // ==========================================================================

  const THEME_KEY = 'abi-docs-theme';
  const DARK_THEME = 'dark';
  const LIGHT_THEME = 'light';

  /**
   * Get the current theme from localStorage or system preference
   */
  function getStoredTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored) return stored;

    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return DARK_THEME;
    }
    return LIGHT_THEME;
  }

  /**
   * Apply theme to document
   */
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
    updateThemeIcon(theme);
  }

  /**
   * Update the theme toggle icon
   */
  function updateThemeIcon(theme) {
    const icon = document.querySelector('.theme-icon');
    if (icon) {
      icon.textContent = theme === DARK_THEME ? '☀️' : '🌙';
    }
  }

  /**
   * Toggle between light and dark themes
   */
  window.toggleTheme = function() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === DARK_THEME ? LIGHT_THEME : DARK_THEME;
    applyTheme(next);
  };

  // ==========================================================================
  // Navigation
  // ==========================================================================

  /**
   * Initialize sidebar navigation
   */
  function initNavigation() {
    // Highlight current page in sidebar
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-list a, .nav-item');

    navLinks.forEach(link => {
      const href = link.getAttribute('href');
      if (href && currentPath.endsWith(href.replace(/^.*\//, ''))) {
        link.classList.add('active');
        // Expand parent section if collapsed
        const section = link.closest('.nav-section');
        if (section) {
          section.classList.add('expanded');
        }
      }
    });
  }

  /**
   * Initialize mobile navigation toggle
   */
  function initMobileNav() {
    // Create mobile menu button
    const navbar = document.querySelector('.nav-container');
    if (!navbar) return;

    const menuButton = document.createElement('button');
    menuButton.className = 'mobile-menu-toggle';
    menuButton.innerHTML = '☰';
    menuButton.setAttribute('aria-label', 'Toggle navigation');
    menuButton.style.display = 'none';

    navbar.appendChild(menuButton);

    menuButton.addEventListener('click', () => {
      const sidebar = document.querySelector('.sidebar');
      if (sidebar) {
        sidebar.classList.toggle('mobile-open');
      }
    });

    // Show/hide based on screen size
    const mediaQuery = window.matchMedia('(max-width: 1024px)');
    function handleResize(e) {
      menuButton.style.display = e.matches ? 'block' : 'none';
    }
    mediaQuery.addEventListener('change', handleResize);
    handleResize(mediaQuery);
  }

  // ==========================================================================
  // Table of Contents
  // ==========================================================================

  /**
   * Initialize table of contents scroll highlighting
   */
  function initTocHighlight() {
    const toc = document.querySelector('.toc');
    if (!toc) return;

    const tocLinks = toc.querySelectorAll('a');
    const headings = [];

    tocLinks.forEach(link => {
      const id = link.getAttribute('href').slice(1);
      const heading = document.getElementById(id);
      if (heading) {
        headings.push({ id, element: heading, link });
      }
    });

    if (headings.length === 0) return;

    function highlightToc() {
      const scrollPosition = window.scrollY + 100;

      let current = headings[0];
      for (const heading of headings) {
        if (heading.element.offsetTop <= scrollPosition) {
          current = heading;
        }
      }

      tocLinks.forEach(link => link.classList.remove('active'));
      current.link.classList.add('active');
    }

    window.addEventListener('scroll', throttle(highlightToc, 100));
    highlightToc();
  }

  // ==========================================================================
  // Code Blocks
  // ==========================================================================

  /**
   * Add copy buttons to code blocks
   */
  function initCodeCopy() {
    const codeBlocks = document.querySelectorAll('pre');

    codeBlocks.forEach(block => {
      const wrapper = document.createElement('div');
      wrapper.className = 'code-wrapper';
      wrapper.style.position = 'relative';

      block.parentNode.insertBefore(wrapper, block);
      wrapper.appendChild(block);

      const button = document.createElement('button');
      button.className = 'copy-button';
      button.textContent = 'Copy';
      button.style.cssText = `
        position: absolute;
        top: 8px;
        right: 8px;
        padding: 4px 8px;
        font-size: 12px;
        background: var(--color-bg-tertiary);
        border: 1px solid var(--color-border);
        border-radius: 4px;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.2s;
      `;

      wrapper.appendChild(button);

      wrapper.addEventListener('mouseenter', () => {
        button.style.opacity = '1';
      });

      wrapper.addEventListener('mouseleave', () => {
        button.style.opacity = '0';
      });

      button.addEventListener('click', async () => {
        const code = block.querySelector('code');
        const text = code ? code.textContent : block.textContent;

        try {
          await navigator.clipboard.writeText(text);
          button.textContent = 'Copied!';
          button.style.background = 'var(--color-success)';
          button.style.color = 'white';

          setTimeout(() => {
            button.textContent = 'Copy';
            button.style.background = '';
            button.style.color = '';
          }, 2000);
        } catch (err) {
          console.error('Failed to copy:', err);
          button.textContent = 'Failed';
        }
      });
    });
  }

  // ==========================================================================
  // Search
  // ==========================================================================

  let searchIndex = null;
  let isSearchOpen = false;

  function getBaseUrl() {
    const link = document.querySelector('link[href*="style.css"]');
    if (link) {
      return link.getAttribute('href').replace(/\/assets\/css\/style\.css$/, '');
    }
    return '';
  }

  /**
   * Initialize search functionality
   */
  async function initSearch() {
    const navbar = document.querySelector('.nav-links');
    if (!navbar) return;

    // Inject Search Button
    const searchBtn = document.createElement('button');
    searchBtn.className = 'search-button';
    searchBtn.innerHTML = `
      <span class="search-icon">🔍</span>
      <span>Search</span>
      <div class="search-keys">
        <span class="search-key">Ctrl</span>
        <span class="search-key">K</span>
      </div>
    `;
    navbar.insertBefore(searchBtn, navbar.firstChild);

    // Inject Modal
    const modalBackdrop = document.createElement('div');
    modalBackdrop.className = 'search-modal-backdrop';
    modalBackdrop.innerHTML = `
      <div class="search-modal" role="dialog" aria-modal="true">
        <div class="search-header">
          <span class="search-icon">🔍</span>
          <input type="text" class="search-input" placeholder="Search documentation..." aria-label="Search">
          <button class="search-close" aria-label="Close search">✕</button>
        </div>
        <div class="search-results"></div>
      </div>
    `;
    document.body.appendChild(modalBackdrop);

    // Event Listeners
    searchBtn.addEventListener('click', openSearch);
    
    modalBackdrop.addEventListener('click', (e) => {
      if (e.target === modalBackdrop) closeSearch();
    });

    modalBackdrop.querySelector('.search-close').addEventListener('click', closeSearch);

    const input = modalBackdrop.querySelector('.search-input');
    input.addEventListener('input', debounce((e) => performSearch(e.target.value), 300));
    
    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        openSearch();
      }
      if (e.key === 'Escape' && isSearchOpen) {
        closeSearch();
      }
    });
  }

  async function openSearch() {
    if (isSearchOpen) return;
    isSearchOpen = true;
    
    const backdrop = document.querySelector('.search-modal-backdrop');
    backdrop.classList.add('open');
    
    const input = backdrop.querySelector('.search-input');
    input.focus();

    if (!searchIndex) {
      try {
        const baseUrl = getBaseUrl();
        // Try multiple paths to be robust
        const paths = [
            `${baseUrl}/search.json`, 
            'search.json', 
            '../search.json', 
            '../../search.json'
        ];
        
        for (const path of paths) {
            try {
                const response = await fetch(path);
                if (response.ok) {
                    searchIndex = await response.json();
                    break;
                }
            } catch (e) { continue; }
        }
        
        if (!searchIndex) console.error('Could not load search index');
      } catch (err) {
        console.error('Failed to load search index:', err);
      }
    }
  }

  function closeSearch() {
    isSearchOpen = false;
    document.querySelector('.search-modal-backdrop').classList.remove('open');
  }

  function performSearch(query) {
    const resultsContainer = document.querySelector('.search-results');
    resultsContainer.innerHTML = '';

    if (!query || !searchIndex) return;

    const normalizedQuery = query.toLowerCase();
    const results = searchIndex.filter(item => {
      const titleMatch = item.title.toLowerCase().includes(normalizedQuery);
      const contentMatch = item.content.toLowerCase().includes(normalizedQuery);
      return titleMatch || contentMatch;
    }).slice(0, 10);

    if (results.length === 0) {
      resultsContainer.innerHTML = '<div class="search-no-results">No results found</div>';
      return;
    }

    const baseUrl = getBaseUrl();
    results.forEach(result => {
      const item = document.createElement('a');
      item.className = 'search-result-item';
      item.href = `${baseUrl}/${result.path}`;
      item.innerHTML = `
        <span class="search-result-title">${result.title}</span>
        <span class="search-result-path">${result.path}</span>
      `;
      item.addEventListener('click', closeSearch);
      resultsContainer.appendChild(item);
    });
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  /**
   * Throttle function calls
   */
  function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }

  /**
   * Debounce function calls
   */
  function debounce(func, wait) {
    let timeout;
    return function(...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }

  // ==========================================================================
  // Smooth Scroll
  // ==========================================================================

  /**
   * Initialize smooth scrolling for anchor links
   */
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function(e) {
        const targetId = this.getAttribute('href').slice(1);
        const target = document.getElementById(targetId);

        if (target) {
          e.preventDefault();
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });

          // Update URL without scrolling
          history.pushState(null, null, `#${targetId}`);
        }
      });
    });
  }

  // ==========================================================================
  // External Links
  // ==========================================================================

  /**
   * Add external link indicators
   */
  function initExternalLinks() {
    const links = document.querySelectorAll('a[href^="http"]');
    const currentHost = window.location.host;

    links.forEach(link => {
      const url = new URL(link.href);
      if (url.host !== currentHost) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
        link.classList.add('external-link');
      }
    });
  }

  // ==========================================================================
  // Initialization
  // ==========================================================================

  function init() {
    // Apply stored theme immediately
    applyTheme(getStoredTheme());

    // Initialize features when DOM is ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initFeatures);
    } else {
      initFeatures();
    }
  }

  function initFeatures() {
    initNavigation();
    initMobileNav();
    initTocHighlight();
    initCodeCopy();
    initSmoothScroll();
    initExternalLinks();
    initSearch();

    console.log('🚀 ABI Documentation initialized');
  }

  // Start initialization
  init();

})();
