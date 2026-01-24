# Service Level Agreement (SLA)

## Overview

This SLA defines the service commitments for ABI commercial support customers.

## Uptime Commitment

For cloud-hosted services (if applicable):
- **Standard:** 99.5% monthly uptime
- **Premium:** 99.9% monthly uptime
- **Enterprise:** 99.95% monthly uptime

## Response Time SLA

### Standard Tier
| Severity | Initial Response | Resolution Target |
|----------|------------------|-------------------|
| Critical | 48 hours | 5 business days |
| High | 72 hours | 10 business days |
| Normal | 5 business days | 20 business days |

### Premium Tier
| Severity | Initial Response | Resolution Target |
|----------|------------------|-------------------|
| Critical | 8 hours | 2 business days |
| High | 24 hours | 5 business days |
| Normal | 48 hours | 10 business days |

### Enterprise Tier
| Severity | Initial Response | Resolution Target |
|----------|------------------|-------------------|
| Critical | 1 hour | 24 hours |
| High | 4 hours | 2 business days |
| Normal | 24 hours | 5 business days |

## Severity Definitions

### Critical (P1)
- Production system completely down
- Data loss or corruption
- Security breach
- No workaround available

### High (P2)
- Major functionality impaired
- Significant performance degradation
- Workaround available but impractical

### Normal (P3)
- Minor functionality issues
- Questions about usage
- Non-critical bugs

### Low (P4)
- Feature requests
- Documentation improvements
- General inquiries

## SLA Credits

If we fail to meet response times:

| Missed SLA | Credit |
|------------|--------|
| 1 occurrence | 5% monthly fee |
| 2 occurrences | 10% monthly fee |
| 3+ occurrences | 25% monthly fee |

Maximum credit: 25% of monthly fee per month.

## Exclusions

SLA does not apply to:
- Scheduled maintenance (48h notice)
- Force majeure events
- Customer-caused issues
- Third-party service outages
- Beta/preview features

## Reporting

- Monthly SLA reports provided
- Real-time status page
- Incident post-mortems for Critical issues
