# AI Log

## 2026-05-25

### Q1
**User**
Adam 실험 이미지 4개와 조건을 공유하며, 기존 SGD 실험 정리 md 파일 아래에 Adam 실험 결과도 추가해달라고 요청함.

**Assistant**
`doc/SGD_EXPERIMENT_SUMMARY.md`에 Adam 실험 조건표, 실험별 관찰, SGD/Adam 비교, Dropout ratio/초기화 비교, 전체 종합 결론을 추가했음을 확인함.

### Q2
**User**
Adam에서 Xavier로 했던 테스트는 실제 Xavier가 아니었으니 문서에서 지워달라고 요청함. 또한 `network.py`에서 He를 Xavier로 바꾸려면 숫자만 `1.0`으로 바꾸면 되는지 질문함.

**Assistant**
`doc/SGD_EXPERIMENT_SUMMARY.md`에서 Adam Xavier 실험 행과 관련 해석을 제거함. 현재 코드 구조에서는 `np.sqrt(2.0 / fan_in)`의 `2.0`을 `1.0`으로 바꾸면 Xavier 방식에 가까워진다고 설명함.

### Q3
**User**
초기값 설정 차이가 드라마틱하지 않아서, He/Xavier 같은 초기화 스케일링을 없애고 랜덤하게 부여하려면 어떻게 하는지 질문함.

**Assistant**
현재 코드도 랜덤값에 He 스케일을 곱하는 방식이며, 스케일링을 없기려면 `np.random.randn(...) * np.sqrt(...)`에서 곱셈 부분을 제거하면 된다고 설명함.

### Q4
**User**
Colab 환경설정 후 작업 폴더가 `/content/WEEK13_AI/WEEK13_AI/...`처럼 반복되어 출력되는데, 테스트가 제대로 이루어지는지 문제인지 질문함.

**Assistant**
환경설정 셀을 현재 폴더 안에서 반복 실행해 중첩 clone과 `os.chdir`가 누적된 상태라고 설명함. `/content`로 돌아간 뒤 기존 폴더를 삭제하고 다시 clone하는 초기화 셀을 안내함.

### Q5
**User**
초기화 스케일링을 없앤 상태에서 드라마틱한 변화를 보려면 SGD로 돌릴지 Adam으로 돌릴지 질문함.

**Assistant**
초기화 문제를 더 선명하게 보려면 먼저 SGD로 돌리는 것을 추천함. Adam은 학습을 보정해 문제를 일부 가릴 수 있고, SGD는 초기값 스케일 변화의 영향을 더 직접적으로 보여준다고 설명함.

### Q6
**User**
Colab 파일 탭에서 코드를 수정했는데, 이 변경이 현재 Colab 환경에서만 바뀐 것인지, 테스트가 변경된 상태로 도는지 질문함.

**Assistant**
Colab에서 수정한 파일은 런타임 내부 복제본만 바뀌며 로컬/깃허브에는 자동 반영되지 않는다고 설명함. 현재 테스트가 수정본을 보는지는 작업 경로와 import된 모듈 경로를 확인해야 한다고 안내함.

### Q7
**User**
SGD에서 초기화 스케일링을 제거한 실험 결과를 공유함. 정확도는 87.77%이고 loss가 천천히 감소한다고 설명하며, 이 내용을 레포트 md 파일에 추가해달라고 요청함.

**Assistant**
`doc/SGD_EXPERIMENT_SUMMARY.md`에 SGD 5번 실험으로 `Random(no scaling)` 결과를 추가함. 실험별 관찰, 초기화 스케일링 제거 비교, 결론, 레포트용 요약, 전체 결론에 학습 속도와 안정성 해석을 반영함.

### Q8
**User**
md 파일에 이미지를 넣을 수 있는지 질문함.

**Assistant**
Markdown에는 `![설명](이미지경로)` 형식으로 이미지를 넣을 수 있으며, GitHub에서 보려면 이미지 파일도 레포 안에 함께 있어야 한다고 설명함.

### Q9
**User**
`doc/test_img` 폴더에 테스트 이미지들을 넣어두었고, `adam+xavier`는 제외한 상태로 레포트 md 파일에 사진들을 모두 넣어달라고 요청함. 마지막 이미지는 `sgd_5`라고 알려줌.

**Assistant**
`doc/SGD_EXPERIMENT_SUMMARY.md`의 각 실험 설명 아래에 `sgd_1`~`sgd_5`, `adam_1`~`adam_3` 이미지를 상대 경로로 추가함. 잘못된 Adam Xavier 이미지인 `adam_4`는 넣지 않았음을 확인함.

### Q10
**User**
제출 양식이 실험 목적, 모델 구조, 학습 설정, 실험 환경, 결과, 회고인데 현재 md 파일이 이 양식대로 작성되었는지 질문함.

**Assistant**
현재 파일은 실험 결과 정리 중심이라 일부 항목은 포함되어 있지만, 제출용 REPORT 형식 1~6번에 딱 맞지는 않는다고 설명함.

### Q11
**User**
제출 양식에 맞게 md 파일을 보기 좋게 수정해달라고 요청함.

**Assistant**
루트의 `REPORT.md`를 제출 양식에 맞춰 재작성함. 실험 목적, 모델 구조, 학습 설정, 실험 환경, 결과, 회고를 정리하고 `sgd_1`~`sgd_5`, `adam_1`~`adam_3` 이미지를 포함함.

### Q12
**User**
수정한 보고서 파일이 어디에 저장되었는지 질문함.

**Assistant**
보고서는 `C:\jungle5\week13\REPORT.md`에 저장되었고, 실험 상세 정리 파일은 `C:\jungle5\week13\doc\SGD_EXPERIMENT_SUMMARY.md`에 있다고 안내함.

### Q13
**User**
사진이 안 나와서 `REPORT.md`를 `doc` 폴더로 옮겼다고 알려줌.

**Assistant**
`REPORT.md`가 `doc` 폴더 안으로 이동되면서 이미지 상대 경로가 `doc/test_img/...`에서 `test_img/...`로 바뀌어야 한다고 확인함. `doc/REPORT.md`의 이미지 경로 8개를 모두 수정함.
