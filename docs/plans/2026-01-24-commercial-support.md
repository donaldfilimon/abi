# Commercial Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish commercial support offerings including SLA, priority support, and custom development services.

**Architecture:** Three integrated components:
1. **SLA Offerings** - Service level agreements for enterprise
2. **Priority Support** - Fast-track assistance channels
3. **Custom Development** - Bespoke feature development

**Tech Stack:** Documentation, process guides, templates

---

## Task 1: Create SLA Framework

**Files:**
- Create: `docs/commercial/README.md`
- Create: `docs/commercial/sla.md`
- Create: `docs/commercial/tiers.md`

**Step 1: Create commercial directory**

```bash
mkdir -p docs/commercial
```

**Step 2: Create commercial overview**

Create `docs/commercial/README.md`:

```markdown
# ABI Commercial Support

## Overview

Enterprise-grade support for organizations using ABI in production.

## Support Tiers

| Tier | Response Time | Coverage | Price |
|------|---------------|----------|-------|
| Community | Best effort | GitHub Issues | Free |
| Professional | 24 hours | Email + GitHub | Contact us |
| Enterprise | 4 hours | Dedicated channel | Contact us |
| Premium | 1 hour | 24/7 phone + chat | Contact us |

## Services

### 📋 SLA Agreements
Guaranteed response times and uptime commitments.
[Learn more](sla.md)

### 🚀 Priority Support
Fast-track issue resolution and direct access to engineers.
[Learn more](priority-support.md)

### 🛠️ Custom Development
Bespoke features and integrations for your needs.
[Learn more](custom-development.md)

## Getting Started

1. **Evaluate** - Review our support tiers
2. **Contact** - Reach out to discuss needs
3. **Agreement** - Sign support contract
4. **Onboard** - Set up support channels
5. **Succeed** - Get help when you need it

## Contact

- Sales: sales@abi.dev
- Support: support@abi.dev
- Phone: [Contact for number]
```

**Step 3: Create SLA document**

Create `docs/commercial/sla.md`:

```markdown
# Service Level Agreements

## Overview

ABI SLAs provide guaranteed response times and service commitments.

## SLA Tiers

### Professional SLA

| Metric | Commitment |
|--------|------------|
| Initial Response | 24 business hours |
| Critical Issues | 8 business hours |
| Severity 1 Resolution | 48 hours |
| Availability Target | 99.5% |
| Support Hours | M-F 9-5 (customer timezone) |

### Enterprise SLA

| Metric | Commitment |
|--------|------------|
| Initial Response | 4 business hours |
| Critical Issues | 2 hours |
| Severity 1 Resolution | 24 hours |
| Availability Target | 99.9% |
| Support Hours | M-F 8-8 (customer timezone) |

### Premium SLA

| Metric | Commitment |
|--------|------------|
| Initial Response | 1 hour |
| Critical Issues | 30 minutes |
| Severity 1 Resolution | 8 hours |
| Availability Target | 99.99% |
| Support Hours | 24/7/365 |

## Severity Definitions

### Severity 1 - Critical
- Production system down
- Data loss or corruption
- Security breach
- No workaround available

### Severity 2 - High
- Major feature unavailable
- Significant performance degradation
- Workaround available but complex

### Severity 3 - Medium
- Minor feature issue
- Performance impact < 20%
- Simple workaround available

### Severity 4 - Low
- Cosmetic issues
- Documentation questions
- Feature requests

## Service Credits

If SLA is not met:

| Uptime | Credit |
|--------|--------|
| 99.0% - 99.5% | 10% |
| 98.0% - 99.0% | 25% |
| 95.0% - 98.0% | 50% |
| < 95.0% | 100% |

## Exclusions

SLA does not apply to:
- Customer-caused issues
- Force majeure events
- Scheduled maintenance
- Beta or preview features
- Issues outside ABI codebase

## Reporting

Monthly SLA reports include:
- Response time metrics
- Resolution time metrics
- Uptime statistics
- Trend analysis
```

**Step 4: Create tiers document**

Create `docs/commercial/tiers.md`:

```markdown
# Support Tiers Comparison

## Feature Matrix

| Feature | Community | Professional | Enterprise | Premium |
|---------|-----------|--------------|------------|---------|
| GitHub Issues | ✅ | ✅ | ✅ | ✅ |
| Email Support | ❌ | ✅ | ✅ | ✅ |
| Priority Queue | ❌ | ✅ | ✅ | ✅ |
| Dedicated Engineer | ❌ | ❌ | ✅ | ✅ |
| Slack/Teams Channel | ❌ | ❌ | ✅ | ✅ |
| Phone Support | ❌ | ❌ | ❌ | ✅ |
| 24/7 Coverage | ❌ | ❌ | ❌ | ✅ |
| Quarterly Reviews | ❌ | ❌ | ✅ | ✅ |
| Architecture Review | ❌ | ❌ | ✅ | ✅ |
| Custom Development | ❌ | Add-on | Included | Included |

## Pricing

### Community
**Free** - Forever

Best for:
- Individual developers
- Open source projects
- Evaluation and testing

### Professional
**Contact for pricing**

Best for:
- Small teams
- Non-critical production
- Standard support needs

### Enterprise
**Contact for pricing**

Best for:
- Large organizations
- Mission-critical systems
- Dedicated support needs

### Premium
**Contact for pricing**

Best for:
- 24/7 operations
- Financial services
- Healthcare
- Government

## Choosing a Tier

### Questions to Ask

1. How critical is ABI to your operations?
2. What are your support hour requirements?
3. Do you need dedicated engineering support?
4. What is your budget for support?

### Recommendations

| Scenario | Recommended Tier |
|----------|------------------|
| Weekend project | Community |
| Startup MVP | Professional |
| Production SaaS | Enterprise |
| Bank/Hospital | Premium |

## Upgrading

Upgrade your tier at any time:
1. Contact sales
2. Review new terms
3. Sign amendment
4. Immediate activation
```

**Step 5: Commit**

```bash
git add docs/commercial/
git commit -m "docs: add SLA framework and support tiers"
```

---

## Task 2: Create Priority Support Documentation

**Files:**
- Create: `docs/commercial/priority-support.md`
- Create: `docs/commercial/support-process.md`
- Create: `docs/commercial/escalation.md`

**Step 1: Create priority support document**

Create `docs/commercial/priority-support.md`:

```markdown
# Priority Support

## Overview

Priority Support provides fast-track assistance for paying customers.

## Features

### Dedicated Queue
- Issues marked as priority
- Guaranteed response times
- Visible SLA tracking

### Direct Access
- Named support engineers
- Direct communication channel
- Skip community queue

### Proactive Support
- Regular check-ins
- Health monitoring
- Upgrade assistance

## How It Works

### 1. Submit Issue
- Use priority support email
- Include account identifier
- Describe issue with severity

### 2. Acknowledgment
- Automatic ticket creation
- SLA timer starts
- Engineer assigned

### 3. Resolution
- Regular updates
- Collaboration on fix
- Verification and closure

## Contact Channels

### Professional Tier
- Email: priority-support@abi.dev
- GitHub: Priority label

### Enterprise Tier
- Email: enterprise-support@abi.dev
- Slack: Dedicated channel
- GitHub: Priority label

### Premium Tier
- Email: premium-support@abi.dev
- Slack: Dedicated channel
- Phone: Dedicated line
- Pager: On-call engineer

## Best Practices

### Before Contacting

1. Check documentation
2. Search existing issues
3. Gather relevant logs
4. Identify severity level

### When Contacting

1. Use correct channel for tier
2. Include account/contract ID
3. Provide reproduction steps
4. Specify environment details

### During Resolution

1. Respond promptly to questions
2. Test proposed solutions
3. Provide feedback on resolution
```

**Step 2: Create support process document**

Create `docs/commercial/support-process.md`:

```markdown
# Support Process

## Issue Lifecycle

```
┌─────────────┐
│   Submit    │
└──────┬──────┘
       ▼
┌─────────────┐
│   Triage    │
└──────┬──────┘
       ▼
┌─────────────┐
│  Investigate │
└──────┬──────┘
       ▼
┌─────────────┐
│   Resolve   │
└──────┬──────┘
       ▼
┌─────────────┐
│   Close     │
└─────────────┘
```

## Stage Details

### Submit
**Owner:** Customer

Actions:
- Create support ticket
- Provide issue details
- Assign severity
- Attach relevant files

### Triage
**Owner:** Support Team

Actions:
- Validate severity
- Assign engineer
- Set SLA targets
- Acknowledge receipt

Timeline: Within SLA initial response

### Investigate
**Owner:** Support Engineer

Actions:
- Reproduce issue
- Analyze root cause
- Develop solution
- Update customer

Timeline: Continuous updates

### Resolve
**Owner:** Support Engineer + Customer

Actions:
- Propose fix
- Customer verification
- Apply solution
- Document resolution

Timeline: Within SLA resolution target

### Close
**Owner:** Support Team

Actions:
- Customer confirmation
- Knowledge base update
- SLA metrics recorded
- Satisfaction survey

## Communication

### Update Frequency

| Severity | Update Interval |
|----------|-----------------|
| Sev 1 | Every 2 hours |
| Sev 2 | Every 4 hours |
| Sev 3 | Daily |
| Sev 4 | As needed |

### Update Content

1. Current status
2. Actions taken
3. Next steps
4. ETA if known
```

**Step 3: Create escalation document**

Create `docs/commercial/escalation.md`:

```markdown
# Escalation Process

## When to Escalate

- SLA breach imminent
- Customer dissatisfaction
- Technical complexity
- Business-critical impact

## Escalation Levels

### Level 1: Support Lead
**Trigger:** 50% SLA time elapsed without progress

Contact:
- Support team lead
- Automatic notification

Actions:
- Additional resources
- Priority adjustment
- Direct involvement

### Level 2: Engineering Manager
**Trigger:** 75% SLA time elapsed or customer request

Contact:
- Engineering manager
- Support lead notification

Actions:
- Engineering team allocation
- Customer communication
- Root cause analysis

### Level 3: Executive
**Trigger:** SLA breach or critical business impact

Contact:
- CTO or designated executive
- All prior levels

Actions:
- All hands response
- Customer executive communication
- Post-mortem planning

## Escalation Matrix

| Severity | L1 Trigger | L2 Trigger | L3 Trigger |
|----------|------------|------------|------------|
| Sev 1 | 30 min | 1 hour | 4 hours |
| Sev 2 | 2 hours | 4 hours | 12 hours |
| Sev 3 | 12 hours | 24 hours | 72 hours |
| Sev 4 | 24 hours | 48 hours | N/A |

## Customer-Initiated Escalation

Customers may request escalation by:
- Email with "ESCALATION" in subject
- Phone call to escalation line
- Slack message with @escalation

## Post-Escalation

After resolution:
1. Root cause analysis
2. Prevention measures
3. Customer follow-up
4. Process improvement
```

**Step 4: Commit**

```bash
git add docs/commercial/
git commit -m "docs: add priority support and escalation process"
```

---

## Task 3: Create Custom Development Framework

**Files:**
- Create: `docs/commercial/custom-development.md`
- Create: `docs/commercial/sow-template.md`
- Create: `docs/commercial/engagement-process.md`

**Step 1: Create custom development document**

Create `docs/commercial/custom-development.md`:

```markdown
# Custom Development Services

## Overview

Custom development services for bespoke ABI features and integrations.

## Service Types

### Feature Development
- New functionality
- API extensions
- Backend implementations
- UI/UX additions

### Integration Services
- Third-party connectors
- Data pipeline setup
- Authentication integration
- Cloud deployment

### Optimization Services
- Performance tuning
- Memory optimization
- GPU utilization
- Scalability improvements

### Consulting Services
- Architecture review
- Best practices guidance
- Training sessions
- Code review

## Engagement Process

1. **Discovery** - Understand requirements
2. **Proposal** - SOW and timeline
3. **Agreement** - Contract signing
4. **Development** - Iterative delivery
5. **Handoff** - Documentation and training

## Pricing Models

### Fixed Price
- Well-defined scope
- Clear deliverables
- Milestone payments

### Time & Materials
- Flexible scope
- Hourly billing
- Monthly invoicing

### Retainer
- Ongoing relationship
- Reserved capacity
- Discounted rates

## Deliverables

All engagements include:
- Source code (Apache 2.0 license)
- Documentation
- Test coverage
- Knowledge transfer

## IP Rights

- Custom code owned by customer
- May be contributed to OSS (optional)
- ABI retains right to similar features

## Getting Started

1. Contact sales@abi.dev
2. Describe your needs
3. Schedule discovery call
4. Receive proposal
5. Begin engagement
```

**Step 2: Create SOW template**

Create `docs/commercial/sow-template.md`:

```markdown
# Statement of Work Template

## Project Information

**Project Name:** [Project Name]
**Client:** [Client Name]
**ABI Contact:** [Contact Name]
**Start Date:** [Date]
**End Date:** [Date]

## Scope of Work

### Objectives
[Describe project objectives]

### Deliverables

| ID | Deliverable | Description | Due Date |
|----|-------------|-------------|----------|
| D1 | [Name] | [Description] | [Date] |
| D2 | [Name] | [Description] | [Date] |

### Out of Scope
[List items explicitly excluded]

## Technical Requirements

### Prerequisites
- [Required by client]

### Environment
- [Development environment]
- [Testing environment]
- [Production environment]

### Technology Stack
- [Technologies to be used]

## Timeline

### Phase 1: [Name]
**Duration:** [X weeks]
- Milestone 1.1: [Description]
- Milestone 1.2: [Description]

### Phase 2: [Name]
**Duration:** [X weeks]
- Milestone 2.1: [Description]
- Milestone 2.2: [Description]

## Resources

### ABI Team
| Role | Name | Allocation |
|------|------|------------|
| Lead Engineer | [Name] | [%] |
| Developer | [Name] | [%] |

### Client Team
| Role | Name | Responsibilities |
|------|------|-----------------|
| Product Owner | [Name] | [Responsibilities] |
| Technical Lead | [Name] | [Responsibilities] |

## Pricing

### Fixed Price
**Total:** $[Amount]

### Payment Schedule
| Milestone | Amount | Due |
|-----------|--------|-----|
| Kickoff | [%] | [Date] |
| Phase 1 Complete | [%] | [Date] |
| Final Delivery | [%] | [Date] |

## Assumptions

1. [Assumption 1]
2. [Assumption 2]

## Acceptance Criteria

Deliverable is accepted when:
1. All functional requirements met
2. Tests pass (>90% coverage)
3. Documentation complete
4. Client sign-off received

## Change Management

Changes to scope require:
1. Written change request
2. Impact assessment
3. Mutual agreement
4. SOW amendment

## Signatures

**Client:**
Name: __________________ Date: __________
Signature: __________________

**ABI:**
Name: __________________ Date: __________
Signature: __________________
```

**Step 3: Create engagement process document**

Create `docs/commercial/engagement-process.md`:

```markdown
# Custom Development Engagement Process

## Phase 1: Discovery (1-2 weeks)

### Activities
- Initial meeting
- Requirements gathering
- Technical assessment
- Scope definition

### Deliverables
- Requirements document
- Technical feasibility report
- High-level estimate

### Your Role
- Share vision and goals
- Provide access to systems
- Identify stakeholders

## Phase 2: Proposal (1 week)

### Activities
- Solution design
- Resource planning
- Timeline development
- Pricing calculation

### Deliverables
- Statement of Work
- Project plan
- Cost estimate

### Your Role
- Review proposal
- Provide feedback
- Negotiate terms

## Phase 3: Agreement (1-2 weeks)

### Activities
- Contract review
- Legal coordination
- Signature collection
- Kickoff planning

### Deliverables
- Signed contract
- Kickoff meeting scheduled
- Access provisioned

### Your Role
- Legal review
- Contract signing
- Team introduction

## Phase 4: Development (Variable)

### Activities
- Iterative development
- Regular demos
- Code reviews
- Testing

### Deliverables
- Working software (incremental)
- Progress reports
- Demo sessions

### Your Role
- Attend sprint reviews
- Provide feedback
- Make decisions promptly

## Phase 5: Delivery (1-2 weeks)

### Activities
- Final testing
- Documentation completion
- Knowledge transfer
- Production deployment

### Deliverables
- Production-ready code
- Complete documentation
- Training materials
- Deployment runbook

### Your Role
- Acceptance testing
- Final sign-off
- Team training

## Phase 6: Support (Ongoing)

### Activities
- Bug fixes (warranty period)
- Ongoing maintenance (optional)
- Feature enhancements (new SOW)

### Deliverables
- Issue resolution
- Maintenance releases
- Enhancement proposals

### Your Role
- Report issues
- Provide feedback
- Plan future work

## Success Factors

### From ABI
- Clear communication
- Quality deliverables
- On-time delivery
- Responsive support

### From Client
- Timely decisions
- Access to resources
- Clear feedback
- Stakeholder availability

## Communication

### Regular Meetings
- Weekly status call
- Sprint demos (bi-weekly)
- Monthly executive summary

### Channels
- Email for formal communication
- Slack for daily questions
- Video calls for meetings

### Escalation
- Project manager first
- Account manager second
- Executive sponsor final
```

**Step 4: Commit**

```bash
git add docs/commercial/
git commit -m "docs: add custom development framework"
```

---

## Task 4: Update ROADMAP

**Files:**
- Modify: `ROADMAP.md` (mark commercial support complete)

**Step 1: Update ROADMAP**

Change:
```markdown
- [ ] Commercial support
  - [ ] SLA offerings
  - [ ] Priority support
  - [ ] Custom development
```

To:
```markdown
- [x] Commercial support - COMPLETE (2026-01-24)
  - [x] SLA offerings (docs/commercial/sla.md)
  - [x] Priority support (docs/commercial/priority-support.md)
  - [x] Custom development (docs/commercial/custom-development.md)
```

**Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "docs: mark commercial support complete in roadmap"
```

---

## Summary

| Component | Files Created | Purpose |
|-----------|---------------|---------|
| SLA Framework | sla.md, tiers.md | Service guarantees |
| Priority Support | priority-support.md, escalation.md | Fast-track assistance |
| Custom Development | custom-development.md, sow-template.md | Bespoke services |

**Total Tasks:** 4
**Estimated Commits:** 4
