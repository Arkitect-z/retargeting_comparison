# Lineage evidence

The graph separates three axes: computational backend, optimization abstraction,
and information content. Edges mean a documented implementation or conceptual
dependency; they do not imply identical evaluation conditions.

```mermaid
flowchart LR
  Mink[Mink backend] --> GMR[GMR]
  Mink --> PM2[ProtoMotions v2.3 retargeting]
  PyRoki[PyRoki backend] --> PM3[ProtoMotions v3 retargeting]
  PHC[PHC-derived preprocessing / FK] --> PM2
  SOCP[Sequential SOCP] --> Omni[OmniRetarget / Holosoma]
  Sparse[Sparse task tracking] --> Dense[Dense body preservation]
  Dense --> Interaction[Interaction preservation]
  Interaction --> Physics[Physics-aware adaptation / control]
  GMR --> Dense
  PM3 --> Dense
  Omni --> Interaction
  MaskedMimic[MaskedMimic controller] --> Physics
  BeyondMimic[BeyondMimic tracker] --> Physics
```

The graph is an evidence map, not a performance ranking.
