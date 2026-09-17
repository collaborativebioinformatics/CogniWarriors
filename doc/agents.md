# Agents Administration Guide

## Purpose
This document specifies which agents should handle specific tasks related to the architecture design and federated learning pipeline. It ensures smooth handovers and maintains consistency when working with the `architecture_design_record.md` file.

## Agent Responsibilities

### Architecture Design Record Management
- **Primary Agent**: Updates and maintains `doc/architecture_design_record.md`
- **Trigger**: Any changes to FL architecture, model assumptions, or design decisions
- **Format**: Follow the existing template with timestamps (newest first), accepted/rejected/outstanding items
- **Example**: See `doc/architecture_design_record.md` for the current record structure

### Federated Learning Pipeline
- **Center Script**: `center.py` - Handle center initialization, worker distribution, FL round coordination
- **Worker Script**: `worker.py` - Handle worker connections, local training, model updates
- **Data Input**: `mri_input.txt` - Structural MRI file paths and configuration
- **Output**: `progress_output.txt` - Training metrics and visualization data

### When to Escalate
- Unclear design decisions in `architecture_design_record.md`
- Conflicting requirements between center and worker implementations
- Performance or scalability issues with current FLARE configuration
- Need to add new data modalities or change input/output formats

### Handover Checklist
1. Review current `architecture_design_record.md` entries
2. Update timestamps and add new design decisions
3. Ensure all accepted/rejected/outstanding items are current
4. Document any arguments/reasons for changes
5. Notify team of architecture updates

## Retention Policy
- **Record Duration**: Archive `architecture_design_record.md` entries after 12 months of inactivity
- **Review Frequency**: Quarterly review of outstanding items and design decisions
- **Backup**: Commit all changes to git with descriptive messages
- **Access Control**: Maintain read access for team members, write access for designated architecture agents
- **Migration**: When archiving, export current state to `doc/architecture_design_record_archive/` directory

### Archive Procedure
1. Create timestamped backup: `cp doc/architecture_design_record.md doc/architecture_design_record_archive/architecture_design_record_YYYYMMDD.md`
2. Clear outstanding items that have been resolved or marked as deferred
3. Update `architecture_design_record.md` with new entry timestamp
4. Commit changes with message: "Archive architecture design record YYYY-MM-DD"

## Commands

### Update Architecture Record
```bash
# Edit the architecture design record
code doc/architecture_design_record.md

# Verify structure
cat doc/architecture_design_record.md | head -20
```

### Run FLARE Pipeline
```bash
# Start center
python center.py --config flare_config.yaml --port 8080

# Start workers (one per center)
python worker.py --center localhost:8080 --worker-id worker_01 --data-dir /data/center01
python worker.py --center localhost:8080 --worker-id worker_02 --data-dir /data/center02
```

### Check Progress
```bash
# Monitor training progress
cat progress_output.txt
```