# Retrieval Scripts

이 폴더는 room retrieval 실험과 측정에 사용하는 canonical CLI script를 관리한다.

30-case suite는 임시 평가 도구이다. 실행·해석 전에 [현재 평가 한계](../../docs/retrieval/README.md#current-evaluation-limitations)를 확인한다.

Canonical scripts:

0. `run_paper_table8.py`
   - 현재 논문 표 8의 Full·객체만·관계만을 실행한다. Full은 객체 점수 동점에서 관측된 공유 벽 또는 단일 개구부 조건을 사용한다. 필요한 지도 증거와 질의별 순위 형식은 [표 8 실행 계약](../../docs/retrieval/paper_table8.md)을 따른다.
   - `run_paper_accuracy.py`는 기존 결합 점수 기준선을 재현한다. [기존 기준선 실행 계약](../../docs/retrieval/paper_accuracy_reference.md)을 따른다.
1. `room_localization_queries.py`
   - Neo4j world graph에 room-localization scenario query를 실행한다.
   - JSON report를 `data/output/*query_report.json`로 저장한다.
2. `profile_room_localization_stages.py`
   - retrieval query의 coarse stage와 component probe 시간을 측정한다.
   - RoomSignature/AnchorGraph 사전 계산이 필요한 병목을 확인하는 용도이다.
3. `evaluate_room_retrieval_metrics.py`
   - 여러 retrieval report를 읽어 Recall@K, Top-1, Top-3, MRR, 평균 ranking time을 집계한다.
4. `run_room_retrieval_suite.py`
   - suite JSON의 여러 view-graph case를 Neo4j에 반복 질의하고 Recall@1/3/5, Top-1/Top-3, MRR, time을 한 번에 집계한다.
   - 최소 단서, multi-view 변형, furnished/sparse 조건을 비교하는 기본 테스트 환경이다.
5. `sync_room_retrieval_docs.py`
   - raw JSON report에서 로컬 검토 문서 `docs/retrieval/raw_json_sync_review_ko.md`를 재생성한다. 이 생성 문서는 Git에 포함하지 않는다.
6. `prepare_reference_inputs.py`
   - 기존 semantic snapshot JSON과 관측 manifest에서 h-hop 부분그래프와 reference vocabulary anchor를 준비한다. DB·matching·성능 실험은 실행하지 않으며 결과 상태는 `not_run`이다. [입력 계약과 사용법](../../docs/retrieval/reference_input_preparation.md)을 따른다.

운영 규칙:

1. 새 명령어 문서는 `scripts/retrieval/...` 경로를 기준으로 작성한다.
2. retrieval 명령은 이 폴더만 사용한다.
3. Codex는 변경에 필요한 pytest를 실행한다. 그 밖의 Python/Conda 처리·Neo4j·실제 benchmark는 해당 작업의 사용자 승인 범위를 따른다.
4. 새 scenario를 추가하면 raw report, suite case/evaluation case, 문서 sync 대상도 함께 갱신한다.
# Experiment environment and diagnostics

See [experiment setup](../../docs/retrieval/experiment_setup.md) for isolated
dependencies, streaming snapshot freeze, hash-checked observation bundles and
bounded GT-free diagnostics. These do not certify Oracle annotations or produce
accuracy metrics.
