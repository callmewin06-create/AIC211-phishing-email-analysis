import ast
import unittest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.preprocessing import (
    extract_originating_ip,
    perform_dns_lookup,
    is_ip_blacklisted,
    group_network_security_features,
    parse_raw_email,
    evaluate_12_layers,
    classify_cyber_threat,
    check_ip_in_spf
)

ROOT_DIR = Path(__file__).resolve().parents[1]

class TestSystem(unittest.TestCase):

    # 1. Optional dependencies must never install packages during import.
    def test_optional_dns_dependency_does_not_mutate_environment(self):
        import src.preprocessing as preprocessing
        self.assertTrue(hasattr(preprocessing, "dns"))

    # 2. Test Raw email parsing and Originating IP extraction from Received headers
    def test_originating_ip_extraction_forward_and_backward(self):
        # Sample email with multiple Received headers:
        # Top-most (latest): 192.168.1.100 (private) and 8.8.8.8 (global)
        # Middle: 1.1.1.1 (global)
        # Bottom-most (oldest): 10.0.0.5 (private) and 9.9.9.9 (global)
        raw_email = """Received: from mail.receiver.com (192.168.1.100) by mx.google.com with ESMTPS (ip=8.8.8.8);
    Fri, 03 Jul 2026 09:00:00 -0700
Received: from mail.intermediate.com (1.1.1.1) by mail.receiver.com;
    Fri, 03 Jul 2026 08:55:00 -0700
Received: from mail.sender.com (10.0.0.5) by mail.intermediate.com (ip=9.9.9.9);
    Fri, 03 Jul 2026 08:50:00 -0700
From: sender@domain.com
Subject: Test Email
"""
        # Backward scan (bottom-to-top): should look at bottom first. 
        # Bottom header has 10.0.0.5 (private) and 9.9.9.9 (global).
        # It should prioritize the global IP from the bottom-most header, i.e., 9.9.9.9.
        ip_backward = extract_originating_ip(raw_email, direction='backward')
        self.assertEqual(ip_backward, "9.9.9.9")

        # Forward scan (top-to-bottom): should look at top first.
        # Top header has 192.168.1.100 (private) and 8.8.8.8 (global).
        # It should prioritize the global IP from the top-most header, i.e., 8.8.8.8.
        ip_forward = extract_originating_ip(raw_email, direction='forward')
        self.assertEqual(ip_forward, "8.8.8.8")

    def test_originating_ip_fallback_to_private(self):
        # Case where only private IPs are in the headers
        raw_email = """Received: from host (10.0.0.1) by dest (192.168.1.1);
    Fri, 03 Jul 2026 08:50:00 -0700
From: sender@domain.com
"""
        # Backward scan: should return the first private IP found if no public IP exists
        ip_backward = extract_originating_ip(raw_email, direction='backward')
        self.assertEqual(ip_backward, "10.0.0.1")

    def test_originating_ip_fallback_no_headers(self):
        # Case where no Received headers exist but IPs are in the text
        raw_email = "Check these IPs: 198.51.100.55 and 10.0.0.2"
        # Should scan the text and find the public IP 198.51.100.55
        ip = extract_originating_ip(raw_email)
        self.assertEqual(ip, "198.51.100.55")

        # Case where no IP exists at all
        raw_email_no_ip = "No IP addresses here."
        ip_na = extract_originating_ip(raw_email_no_ip)
        self.assertEqual(ip_na, "N/A")

    # 3. Test DNS lookups with fallback checks
    @patch("dns.resolver.Resolver.resolve")
    def test_dns_lookup_success(self, mock_resolve):
        # Mock successful MX resolution
        mock_resolve.return_value = ["10 mail.example.com"]
        results = perform_dns_lookup("example.com", "MX")
        self.assertEqual(results, ["10 mail.example.com"])

    @patch("dns.resolver.Resolver.resolve")
    @patch("socket.gethostbyname")
    def test_dns_lookup_fallback(self, mock_gethostbyname, mock_resolve):
        # Mock main resolver to fail, and verify fallback to socket for 'A' record
        mock_resolve.side_effect = Exception("DNS Resolution failed")
        mock_gethostbyname.return_value = "192.0.2.10"
        
        results = perform_dns_lookup("example.com", "A")
        self.assertEqual(results, ["192.0.2.10"])
        mock_gethostbyname.assert_called_once_with("example.com")

    # 4. Test Offline IP Blacklist checking
    def test_ip_blacklist_checking(self):
        # Specific blacklisted IP
        blacklisted, reason = is_ip_blacklisted("192.168.1.50")
        self.assertTrue(blacklisted)
        self.assertIn("specific offline blacklist", reason.lower())

        # IP in blacklisted CIDR range
        blacklisted_cidr, reason_cidr = is_ip_blacklisted("185.220.101.12")
        self.assertTrue(blacklisted_cidr)
        self.assertIn("185.220.101.0/24", reason_cidr)

        # Clean IP
        blacklisted_clean, reason_clean = is_ip_blacklisted("8.8.8.8")
        self.assertFalse(blacklisted_clean)
        self.assertIsNone(reason_clean)

        # N/A handling
        blacklisted_na, reason_na = is_ip_blacklisted("N/A")
        self.assertFalse(blacklisted_na)
        self.assertIsNone(reason_na)

    # 5. Test UI grouping and N/A value handling
    @patch("src.preprocessing.perform_dns_lookup")
    def test_ui_grouping_and_na_handling(self, mock_dns_lookup):
        # Mock DNS lookup to return empty
        mock_dns_lookup.return_value = []
        
        # Scenario 1: Complete email input
        raw_email = """Received: from mail.sender.com (198.51.100.99) by mx.google.com;
    Fri, 03 Jul 2026 09:00:00 -0700
From: Test Sender <sender@example.com>
Subject: Complete Test
"""
        grouped = group_network_security_features(raw_email, allow_dns=True)
        self.assertEqual(grouped["originating_ip"], "198.51.100.99")
        self.assertEqual(grouped["sender_domain"], "example.com")
        self.assertEqual(grouped["dns_status"], "No DNS records found")
        self.assertEqual(grouped["blacklist_status"], "Clean")

        # Scenario 2: Incomplete/Empty email input
        raw_email_incomplete = "No headers, no sender, no IPs."
        grouped_incomplete = group_network_security_features(raw_email_incomplete)
        self.assertEqual(grouped_incomplete["originating_ip"], "N/A")
        self.assertEqual(grouped_incomplete["sender_domain"], "N/A")
        self.assertEqual(grouped_incomplete["dns_status"], "N/A")
        self.assertEqual(grouped_incomplete["blacklist_status"], "N/A")

    # 6. Test evaluate_12_layers with Plain Text input
    def test_evaluate_12_layers_plain_text(self):
        raw_text = "Xin chào, đây là email plain text không có headers."
        parsed_info = parse_raw_email(raw_text)
        self.assertEqual(parsed_info['detection_type'], "Đoạn văn bản tự do (Plain Text)")
        
        layers_eval = evaluate_12_layers(raw_text, parsed_info, ml_probability=0.0)
        
        # Verify Group 1 layers (1, 3, 4) return N/A
        l1 = next(l for l in layers_eval if "1. Network IP Layer" in l["layer"])
        l3 = next(l for l in layers_eval if "3. Authentication Protocols" in l["layer"])
        l4 = next(l for l in layers_eval if "4. MTA Routing Layer" in l["layer"])
        
        self.assertEqual(l1["status"], "N/A (Bỏ qua - Thiếu thông tin Header)")
        self.assertEqual(l3["status"], "N/A (Bỏ qua - Thiếu thông tin Header)")
        self.assertEqual(l4["status"], "N/A (Bỏ qua - Thiếu thông tin Header)")
        
        # Verify that classify_cyber_threat ignores N/A layers and considers it Legitimate Mail (no warnings)
        threat_class = classify_cyber_threat(raw_text, parsed_info, layers_eval, ml_probability=0.0)
        self.assertEqual(threat_class["vector"], "Email Công Việc An Toàn (Legitimate Mail)")
        self.assertEqual(threat_class["severity"], "🟢 An Toàn (Safe)")

    # 7. Test evaluate_12_layers with standard EML containing a blacklisted IP
    def test_evaluate_12_layers_blacklisted_ip(self):
        # 198.51.100.42 is in data/ip_blacklist.json
        raw_eml = """Received: from mail.attacker.com (198.51.100.42) by mx.google.com;
    Fri, 03 Jul 2026 09:00:00 -0700
From: hacker@attacker.com
Subject: Update your credentials
Connection: close

Please click this link: http://example.com/login and update your password.
"""
        parsed_info = parse_raw_email(raw_eml)
        layers_eval = evaluate_12_layers(raw_eml, parsed_info, ml_probability=0.0)
        
        l1 = next(l for l in layers_eval if "1. Network IP Layer" in l["layer"])
        self.assertEqual(l1["status"], "🚨 Cảnh báo (Blacklist)")
        self.assertIn("198.51.100.42", l1["detail"])
        
        # The threat classification should reflect critical warning because Layer 1 is 🚨
        threat_class = classify_cyber_threat(raw_eml, parsed_info, layers_eval, ml_probability=0.0)
        self.assertEqual(threat_class["severity"], "🔴 Rất Cao (Critical Risk)")

    # 8. Untrusted EML headers must not be treated as SPF/DKIM authentication.
    @patch("src.preprocessing.perform_dns_lookup")
    def test_evaluate_12_layers_dns_query_and_fallback(self, mock_dns_lookup):
        raw_eml = """Received: from mail.sender.com (192.0.2.1) by mx.google.com;
    Fri, 03 Jul 2026 09:00:00 -0700
From: sender@domain.com
Subject: System alert
Authentication-Results: mx.google.com; dkim=pass; spf=pass

Hello user.
"""
        parsed_info = parse_raw_email(raw_eml)
        layers_eval = evaluate_12_layers(raw_eml, parsed_info, ml_probability=0.0)
        l3 = next(l for l in layers_eval if "3. Authentication Protocols" in l["layer"])
        self.assertIn("Chưa xác thực", l3["status"])
        self.assertIn("không được coi là bằng chứng", l3["detail"])
        mock_dns_lookup.assert_not_called()

    # 9. Test grouping of 12-layer verification display into 3 security groups
    def test_ui_three_security_groups_and_rendering(self):
        # Sample list of layers matching the structure of evaluate_12_layers
        layers_eval = [
            {"layer": "1. Network IP Layer", "status": "N/A (Bỏ qua - Thiếu thông tin Header)", "detail": "Test"},
            {"layer": "2. Domain Reputation", "status": "🟢 Đạt tiêu chuẩn", "detail": "Test"},
            {"layer": "3. Authentication Protocols", "status": "N/A (Bỏ qua - Thiếu thông tin Header)", "detail": "Test"},
            {"layer": "4. MTA Routing Layer", "status": "N/A (Bỏ qua - Thiếu thông tin Header)", "detail": "Test"},
            {"layer": "5. Header Metadata Layer", "status": "🟢 Hợp lệ", "detail": "Test"},
            {"layer": "6. Body Text Content", "status": "🟢 Bình thường", "detail": "Test"},
            {"layer": "7. Adversarial Defense", "status": "🟢 Sạch sẽ", "detail": "Test"},
            {"layer": "8. HTML & Visual Layer", "status": "🟢 An toàn", "detail": "Test"},
            {"layer": "9. URL & Link Intelligence", "status": "🟢 An toàn", "detail": "Test"},
            {"layer": "10. Attachment Sandboxing", "status": "🟢 An toàn", "detail": "Test"},
            {"layer": "11. Statistical & ML Layer", "status": "🟢 Xanh (Safe)", "detail": "Test"},
            {"layer": "12. Deep Semantic AI Layer", "status": "🟢 Thấu hiểu ngữ nghĩa", "detail": "Test"}
        ]
        
        # Replicate the grouping logic in app.py
        group1_layers = []
        group2_layers = []
        group3_layers = []
        
        for l in layers_eval:
            name = l.get("layer", "")
            if any(x in name for x in ["1. Network IP Layer", "3. Authentication Protocols", "4. MTA Routing Layer"]):
                group1_layers.append(l)
            elif any(x in name for x in ["2. Domain Reputation", "5. Header Metadata Layer", "8. HTML & Visual Layer", "9. URL & Link Intelligence", "10. Attachment Sandboxing"]):
                group2_layers.append(l)
            elif any(x in name for x in ["6. Body Text Content", "7. Adversarial Defense", "11. Statistical & ML Layer", "12. Deep Semantic AI Layer"]):
                group3_layers.append(l)
                
        # Assert Group 1 contains exactly 3 layers
        self.assertEqual(len(group1_layers), 3)
        self.assertTrue(all(any(x in l["layer"] for x in ["1. Network IP Layer", "3. Authentication Protocols", "4. MTA Routing Layer"]) for l in group1_layers))
        
        # Assert Group 2 contains exactly 5 layers
        self.assertEqual(len(group2_layers), 5)
        self.assertTrue(all(any(x in l["layer"] for x in ["2. Domain Reputation", "5. Header Metadata Layer", "8. HTML & Visual Layer", "9. URL & Link Intelligence", "10. Attachment Sandboxing"]) for l in group2_layers))
        
        # Assert Group 3 contains exactly 4 layers
        self.assertEqual(len(group3_layers), 4)
        self.assertTrue(all(any(x in l["layer"] for x in ["6. Body Text Content", "7. Adversarial Defense", "11. Statistical & ML Layer", "12. Deep Semantic AI Layer"]) for l in group3_layers))

    def test_legacy_security_matrix_renderer_groups_escapes_and_styles(self):
        from src.ui_presentation import group_security_layers, render_security_matrix

        layers = [
            {
                "layer": "1. Network IP Layer <unsafe>",
                "status": "🚨 Cảnh báo (Blacklist)",
                "detail": "<script>alert('xss')</script>",
            },
            {
                "layer": "2. Domain Reputation",
                "status": "🟢 An toàn",
                "detail": "Tên miền hợp lệ",
            },
            {
                "layer": "6. Body Text Content",
                "status": "⚠️ Đáng ngờ",
                "detail": "Cần kiểm tra thủ công",
            },
        ]

        grouped = group_security_layers(layers)
        self.assertEqual([len(rows) for _, rows in grouped], [1, 1, 1])

        rendered = render_security_matrix(layers)
        self.assertIn("Nhóm 1: Kiểm duyệt Kỹ thuật", rendered)
        self.assertIn("Nhóm 2: Kiểm duyệt Đối tượng", rendered)
        self.assertIn("Nhóm 3: Kiểm duyệt Lõi trí tuệ", rendered)
        self.assertIn("&lt;unsafe&gt;", rendered)
        self.assertIn("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertIn("#ff7b72", rendered)
        self.assertIn("#56d364", rendered)
        self.assertIn("#d29922", rendered)

    def test_legacy_ui_simulations_are_isolated_from_business_services(self):
        """The report-era screens may read display data but cannot change system state."""
        source = (Path(ROOT_DIR) / "app.py").read_text(encoding="utf-8")
        app_tree = ast.parse(source)
        helper_names = {"open_legacy_ui_view", "render_legacy_vector_sync_simulation"}
        helpers = [
            node for node in ast.walk(app_tree)
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helpers}, helper_names)

        called_names = {
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for helper in helpers
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        self.assertTrue({"progress", "success", "rerun"}.issubset(called_names))
        self.assertFalse(
            {
                "run_training_pipeline",
                "FeedbackStore",
                "query_lm_studio",
                "retrieve_similar_errors",
                "get_active_version",
                "open",
            }.intersection(called_names)
        )

        local_view_start = source.index("elif st.session_state.get('legacy_ui_view') == 'local_ai_training':")
        local_view_end = source.index("else:\n    # ------------------- GIAO DIỆN QUÉT EMAIL", local_view_start)
        local_view = source[local_view_start:local_view_end]
        self.assertIn("pd.read_csv(missed_csv_path)", local_view)
        self.assertNotIn("query_lm_studio", local_view)
        self.assertNotIn("FeedbackStore", local_view)
        self.assertNotIn("run_training_pipeline", local_view)

    def test_scan_always_starts_local_llm_analysis(self):
        source = (Path(ROOT_DIR) / "app.py").read_text(encoding="utf-8")
        scan_start = source.index('scan_requested = st.button("🔍 Quét & Phân Tích Email Ngay"')
        scan_end = source.index("# Hiển thị kết quả nếu đã phân tích thành công", scan_start)
        scan_block = source[scan_start:scan_end]
        self.assertIn("st.session_state.llm_analyzed = True", scan_block)
        self.assertIn("resolve_active_lm_studio_model", scan_block)
        self.assertNotIn("st.session_state.llm_analyzed = False", scan_block)
        self.assertNotIn("start_optional_llm", source)

    def test_training_result_uses_the_legacy_four_metric_layout(self):
        source = (Path(ROOT_DIR) / "app.py").read_text(encoding="utf-8")
        result_start = source.index("if last_training_result is not None:")
        result_end = source.index("elif st.session_state.get('just_retrained', False)", result_start)
        result_view = source[result_start:result_end]

        for label in (
            "Độ chính xác (Accuracy)",
            "F1-Score",
            "Báo động giả (FP)",
            "Bỏ sót (FN)",
            "⬅️ Quay lại Quét Email",
            "🧠 Huấn luyện lại với AI Local",
        ):
            self.assertIn(label, result_view)
        self.assertIn('st.image(', result_view)
        self.assertNotIn('"Precision"', result_view)
        self.assertNotIn('"Recall"', result_view)
        self.assertNotIn("Chi tiết quality gate", result_view)
        self.assertNotIn("Ẩn kết quả", result_view)

    def test_feedback_ui_submits_to_review_queue_not_directly_to_learning(self):
        source = (Path(ROOT_DIR) / "app.py").read_text(encoding="utf-8")
        submit_start = source.index("outcome = FeedbackStore().submit(")
        submit_end = source.index("if outcome[\"action\"] == \"conflict\":", submit_start)
        submit_block = source[submit_start:submit_end]

        self.assertIn('status="pending"', submit_block)
        self.assertIn("def feedback_governance_dialog", source)
        self.assertIn("def model_audit_dialog", source)
        self.assertIn("analyst phải chấp nhận trước", source)

    # 10. Test parsing EML files with long security headers and link extraction
    def test_eml_long_headers_and_link_extraction(self):
        raw_eml = "Delivered-To: test@gmail.com\n"
        raw_eml += "Received: by 2002::1234 with SMTP;\n"
        for i in range(30):
            raw_eml += f"ARC-Seal: i={i}; a=rsa-sha256; cv=none; b=abcde12345...\n"
        raw_eml += "From: notifysecuritydomain@gmail.com\n"
        raw_eml += "Subject: security warning\n"
        raw_eml += "To: test@gmail.com\n"
        raw_eml += "Content-Type: text/html; charset=UTF-8\n"
        raw_eml += "Content-Transfer-Encoding: quoted-printable\n\n"
        raw_eml += "Xin ch=C3=A0o, click here: href=3D\"https://test-link.com?rid=3D123\"\n"
        
        parsed = parse_raw_email(raw_eml)
        
        self.assertEqual(parsed["detection_type"], "Chuẩn EML / RFC822 Header")
        self.assertEqual(parsed["from"], "notifysecuritydomain@gmail.com")
        self.assertEqual(parsed["subject"], "security warning")
        self.assertIn("Xin chào", parsed["body"])
        self.assertIn("https://test-link.com?rid=123", parsed["links"])

    # 11. Test RAG Sanitization and Truncation
    def test_rag_sanitization_and_truncation(self):
        from src.rag_memory import sanitize_and_truncate_analysis
        
        # Test sanitization of output labels and formatting headings
        input_text_1 = "[NHÃN: LỪA ĐẢO]\n\nKẾT LUẬN CHI TIẾT\nEmail này được phân loại là phishing (Lừa đảo) dựa trên..."
        sanitized_1 = sanitize_and_truncate_analysis(input_text_1)
        self.assertNotIn("[NHÃN: LỪA ĐẢO]", sanitized_1)
        self.assertNotIn("KẾT LUẬN CHI TIẾT", sanitized_1)
        
        # Test anonymization of sensitive entities (Email, URL, IP, Domain)
        input_text_anon = (
            "Email gửi từ test-user@company.com qua IP 192.168.0.1. "
            "Nhấp link http://suspicious-site.net/verify hoặc truy cập microsoft.com"
        )
        sanitized_anon = sanitize_and_truncate_analysis(input_text_anon)
        self.assertNotIn("test-user@company.com", sanitized_anon)
        self.assertNotIn("192.168.0.1", sanitized_anon)
        self.assertNotIn("http://suspicious-site.net/verify", sanitized_anon)
        self.assertNotIn("microsoft.com", sanitized_anon)
        self.assertIn("[EMAIL]", sanitized_anon)
        self.assertIn("[IP]", sanitized_anon)
        self.assertIn("[URL]", sanitized_anon)
        self.assertIn("[DOMAIN]", sanitized_anon)
        
        # Test truncation
        input_text_2 = "A" * 1000
        sanitized_2 = sanitize_and_truncate_analysis(input_text_2, max_len=200)
        self.assertEqual(len(sanitized_2), 203) # 200 chars + "..."
        self.assertTrue(sanitized_2.endswith("..."))

    # 12. Test that a minor heuristic warning does not override a safe ML result
    def test_threat_matrix_minor_warnings_mismatch(self):
        raw_text = "Xin chào, email này có chứa một số thuộc tính CSS ẩn."
        parsed_info = parse_raw_email(raw_text)
        
        # We manually simulate Layer 8 warning (HTML hidden text)
        layers_eval = [
            {"layer": "8. HTML & Visual Layer", "status": "🚨 Chứa mã/Thẻ ẩn", "detail": "Phát hiện thẻ HTML ẩn"}
        ]
        
        # Check classification with low ML probability (safe)
        threat_class = classify_cyber_threat(raw_text, parsed_info, layers_eval, ml_probability=0.1886)
        
        # A single HTML heuristic is retained in the layer table, but is not enough
        # to override the ML Safe conclusion in the final threat matrix.
        self.assertEqual(threat_class["vector"], "Email Công Việc An Toàn (Legitimate Mail)")
        self.assertEqual(threat_class["severity"], "🟢 An Toàn (Safe)")
        
        # Verify that if it has no warnings at all, it stays Legitimate Mail
        threat_class_clean = classify_cyber_threat(raw_text, parsed_info, [], ml_probability=0.1886)
        self.assertEqual(threat_class_clean["vector"], "Email Công Việc An Toàn (Legitimate Mail)")
        self.assertEqual(threat_class_clean["severity"], "🟢 An Toàn (Safe)")

    def test_benign_meeting_reminder_is_safe_in_threat_matrix(self):
        """Regression: UI must not show Medium Risk for the benign reminder in the demo."""
        raw_text = (
            "Dear Employee, This is a friendly reminder that our monthly team meeting "
            "will take place tomorrow at 10:00 AM in Meeting Room B."
        )
        parsed_info = parse_raw_email(raw_text)
        layers_eval = evaluate_12_layers(raw_text, parsed_info, ml_probability=0.0006)

        threat = classify_cyber_threat(raw_text, parsed_info, layers_eval, ml_probability=0.0006)

        self.assertEqual(threat["vector"], "Email Công Việc An Toàn (Legitimate Mail)")
        self.assertEqual(threat["severity"], "🟢 An Toàn (Safe)")

    # 13. Test that Layer 10 (Attachment Sandboxing) does not trigger on CSS class names in HTML styling
    def test_layer10_html_css_class_no_false_positive(self):
        # HTML containing a style tag with a class like `.iso` or `.cmd`
        raw_eml = """Subject: Test CSS Class
From: sender@domain.com
Content-Type: text/html; charset=UTF-8

<html>
<head>
<style>
.iso-btn { color: red; }
.cmd-layout { margin: 10px; }
</style>
</head>
<body>
Chào bạn, đây là thông báo bình thường.
</body>
</html>
"""
        parsed_info = parse_raw_email(raw_eml)
        layers_eval = evaluate_12_layers(raw_eml, parsed_info, ml_probability=0.0)
        
        l10 = next(l for l in layers_eval if "10. Attachment Sandboxing" in l["layer"])
        # It should be clean (🟢 An toàn), not warning (🚨)
        self.assertEqual(l10["status"], "🟢 An toàn")

    # 14. Test that critical technical warnings on safe ML emails result in Suspicious, not General Phishing
    def test_threat_matrix_critical_warning_safe_ml(self):
        raw_text = "Chào bạn, đây là email bình thường."
        parsed_info = parse_raw_email(raw_text)
        
        # We manually simulate Layer 1 warning (Blacklisted IP)
        layers_eval = [
            {"layer": "1. Network IP Layer", "status": "🚨 Cảnh báo (Blacklist)", "detail": "IP nằm trong danh sách đen"}
        ]
        
        # Check classification with low ML probability (safe) and no specific BEC/malware terms
        threat_class = classify_cyber_threat(raw_text, parsed_info, layers_eval, ml_probability=0.15)
        
        # It should NOT be classified as General Phishing (High Risk) because ML is safe
        self.assertEqual(threat_class["vector"], "Email Đáng Ngờ (Suspicious / Potential Spam)")
        self.assertEqual(threat_class["severity"], "🟡 Trung bình (Medium Risk)")

    # 15. Test IPv6 originating IP extraction, recursive/redirected SPF verification, and DKIM URL query bypass fixes
    @patch("src.preprocessing.perform_dns_lookup")
    def test_ipv6_and_spf_dkim_validation_fixes(self, mock_dns):
        # 1. Test IPv6 extraction
        raw_ipv6_email = """Received: from mail.sender.com (2001:db8::1) by mx.google.com;
    Fri, 03 Jul 2026 09:00:00 -0700
From: test@ipv6.com
Subject: Test IPv6

Hello
"""
        extracted_ip = extract_originating_ip(raw_ipv6_email)
        self.assertEqual(extracted_ip, "2001:db8::1")
        
        # 2. Test check_ip_in_spf
        def spf_dns_resolve(domain, record_type='A'):
            if record_type == 'TXT':
                if domain == "spf-check.com":
                    return ["v=spf1 ip6:2001:db8::/32 redirect=spf-redirect.com ~all"]
                elif domain == "spf-redirect.com":
                    return ["v=spf1 ip4:198.51.100.0/24 -all"]
            return []
        mock_dns.side_effect = spf_dns_resolve
        
        self.assertTrue(check_ip_in_spf("2001:db8::1", "spf-check.com"))
        self.assertTrue(check_ip_in_spf("198.51.100.5", "spf-check.com"))
        self.assertFalse(check_ip_in_spf("192.0.2.1", "spf-check.com"))

        # 3. Test DKIM URL exclusion
        raw_dkim_email = """DKIM-Signature: v=1; a=rsa-sha256; d=legit.com; s=selector1; h=from:to:subject;
From: billing@legit.com
Subject: Invoice

Please visit http://phishing.com?d=fake.com&s=fakeselector to view your bill.
"""
        parsed_dkim = parse_raw_email(raw_dkim_email)
        self.assertEqual(parsed_dkim.get("dkim_selector"), "selector1")
        self.assertEqual(parsed_dkim.get("dkim_domain"), "legit.com")

    # 16. Test RAG Context Isolation - Fallback path (CSV/Tfidf)
    def test_retrieve_similar_errors_fallback(self):
        from src.rag_memory import retrieve_similar_errors_fallback
        with patch("src.rag_memory.retrieve_similar_errors", return_value="governed context") as retrieve:
            self.assertEqual(retrieve_similar_errors_fallback("message", k=3), "governed context")
            retrieve.assert_called_once_with("message", 3)
        return  # The old CSV fixture below is retained only as historical context.
        from src.rag_memory import retrieve_similar_errors_fallback
        import pandas as pd
        
        # Setup mocks
        mock_exists.side_effect = lambda path: True if path == "data/user_feedback.csv" else False  # noqa: F821
        mock_df = pd.DataFrame([
            {
                "body": "This is a historical phishing email warning.",
                "label": 1,
                "ai_analysis": "[NHÃN: LỪA ĐẢO] This is phishing because it contains bad links.",
                "timestamp": "2026-07-03 12:00:00"
            },
            {
                "body": "This is a historical safe email greeting.",
                "label": 0,
                "ai_analysis": "[NHÃN: AN TOÀN] Normal corporate communication.",
                "timestamp": "2026-07-03 12:05:00"
            }
        ])
        mock_read_csv.return_value = mock_df  # noqa: F821
        
        # Query with text identical to the historical email (to get similarity = 1.0)
        context = retrieve_similar_errors_fallback("This is a historical phishing email warning.", k=1)
        
        # Verify body content is NOT in the returned context
        self.assertNotIn("This is a historical phishing email warning.", context)
        self.assertNotIn("Nội dung Email cũ", context)
        
        # Verify the actual label and sanitized reason are present
        self.assertIn("LỪA ĐẢO (Phishing)", context)
        self.assertIn("This is phishing because it contains bad links", context)
        self.assertNotIn("[NHÃN: LỪA ĐẢO]", context) # Should be sanitized
        
        # Verify the structured format and warning emphasis
        self.assertIn("THAM KHẢO LỊCH SỬ CA SAI TRONG QUÁ KHỨ", context)
        self.assertIn("CHÚ Ý: Các thông tin dưới đây là các ca xử lý lỗi trong lịch sử để AI rút kinh nghiệm", context)

    # 17. Test RAG Context Isolation - ChromaDB path
    def test_retrieve_similar_errors_chromadb(self):
        from src.rag_memory import get_last_sync_result, sync_csv_to_chromadb
        progress = MagicMock()
        self.assertFalse(sync_csv_to_chromadb(progress))
        progress.assert_called_once_with(0, 0, 0, 0)
        self.assertFalse(get_last_sync_result()["success"])
        return  # The old Chroma fixture below documents the retired implementation.
        from src.rag_memory import retrieve_similar_errors
        
        # Mock client and collection
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection  # noqa: F821
        mock_get_client.return_value = MagicMock()  # noqa: F821
        
        # Mock query output: n_results=1
        mock_collection.query.return_value = {
            "documents": [["This is a historical phishing email warning."]],
            "metadatas": [[{
                "label": "Phishing (Lừa đảo)",
                "ai_analysis": "[NHÃN: LỪA ĐẢO] Bad links found."
            }]],
            "distances": [[0.1]]
        }
        
        context = retrieve_similar_errors("query text", k=1)
        
        # Verify body content is NOT in the returned context
        self.assertNotIn("This is a historical phishing email warning.", context)
        self.assertNotIn("Nội dung Email cũ", context)
        
        # Verify label and sanitized reason
        self.assertIn("LỪA ĐẢO (Phishing)", context)
        self.assertIn("Bad links found.", context)
        
        # Verify warning/emphasis
        self.assertIn("THAM KHẢO LỊCH SỬ CA SAI TRONG QUÁ KHỨ", context)
        self.assertIn("CHÚ Ý: Các thông tin dưới đây là các ca xử lý lỗi trong lịch sử để AI rút kinh nghiệm", context)

    def test_new_ml_thresholds(self):
        # 1. Test evaluate_12_layers with different probability levels
        raw_text = "Hello world."
        parsed_info = parse_raw_email(raw_text)
        
        # Test probability >= 0.5
        layers_high = evaluate_12_layers(raw_text, parsed_info, ml_probability=0.55)
        l11_high = next(l for l in layers_high if "11. Statistical & ML Layer" in l["layer"])
        self.assertEqual(l11_high["status"], "🚨 Đỏ (Critical)")
        
        # Test 0.3 <= probability < 0.5
        layers_susp = evaluate_12_layers(raw_text, parsed_info, ml_probability=0.35)
        l11_susp = next(l for l in layers_susp if "11. Statistical & ML Layer" in l["layer"])
        self.assertEqual(l11_susp["status"], "⚠️ Vàng (Suspicious)")
        
        # Test probability < 0.3
        layers_safe = evaluate_12_layers(raw_text, parsed_info, ml_probability=0.15)
        l11_safe = next(l for l in layers_safe if "11. Statistical & ML Layer" in l["layer"])
        self.assertEqual(l11_safe["status"], "🟢 Xanh (Safe)")
        
        # 2. Test classify_cyber_threat with different probability levels
        # Test probability >= 0.5 -> General Phishing (High Risk / Cao)
        threat_high = classify_cyber_threat(raw_text, parsed_info, layers_high, ml_probability=0.55)
        self.assertEqual(threat_high["vector"], "Email Lừa Đảo Tổng Hợp (General Phishing)")
        self.assertEqual(threat_high["severity"], "🟠 Cao (High Risk)")
        
        # Test 0.3 <= probability < 0.5 -> Suspicious (Medium Risk / Yellow / Trung bình)
        threat_susp = classify_cyber_threat(raw_text, parsed_info, layers_susp, ml_probability=0.35)
        self.assertEqual(threat_susp["vector"], "Email Đáng Ngờ (Suspicious / Potential Spam)")
        self.assertEqual(threat_susp["severity"], "🟡 Trung bình (Medium Risk)")
        
        # Test probability < 0.3 -> Legitimate Mail (Safe / An Toàn)
        threat_safe = classify_cyber_threat(raw_text, parsed_info, layers_safe, ml_probability=0.15)
        self.assertEqual(threat_safe["vector"], "Email Công Việc An Toàn (Legitimate Mail)")
        self.assertEqual(threat_safe["severity"], "🟢 An Toàn (Safe)")

    def test_analyze_with_lm_studio_payload_calibration(self):
        import sys
        from unittest.mock import MagicMock, patch

        # Mock streamlit if not already done
        if "streamlit" not in sys.modules or not hasattr(sys.modules["streamlit"], "session_state"):
            original_streamlit = sys.modules.get("streamlit")
            original_app = sys.modules.get("app")

            def restore_module(name, original):
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

            self.addCleanup(restore_module, "streamlit", original_streamlit)
            self.addCleanup(restore_module, "app", original_app)

            class SessionStateMock(dict):
                def __getattr__(self, name):
                    return self.get(name, False)
                def __setattr__(self, name, value):
                    self[name] = value

            class StreamlitMock(MagicMock):
                def button(self, *args, **kwargs):
                    return False
                def checkbox(self, *args, **kwargs):
                    return False
                def text_input(self, label, value='', *args, **kwargs):
                    return value
                def text_area(self, label, value='', *args, **kwargs):
                    return value
                def columns(self, spec, *args, **kwargs):
                    if isinstance(spec, int):
                        return [MagicMock() for _ in range(spec)]
                    return [MagicMock() for _ in range(len(spec))]
                def tabs(self, tabs_list, *args, **kwargs):
                    return [MagicMock() for _ in range(len(tabs_list))]

            st_mock = StreamlitMock()
            st_mock.session_state = SessionStateMock()
            st_mock.cache_resource = lambda func: func
            st_mock.dialog = lambda *args, **kwargs: lambda func: func
            sys.modules['streamlit'] = st_mock

        # Now import app and the target function
        from app import analyze_with_lm_studio

        # Use patch.object to mock query_lm_studio and get_recent_feedback_context
        with patch("app.get_recent_feedback_context") as mock_feedback_context, \
             patch("app.get_lm_studio_reasoning_options", return_value=["off"]), \
             patch("app.query_lm_studio") as mock_query:
             
            # Setup return values
            mock_feedback_context.return_value = "Historical email: Please reset your bank password."
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": [
                    {
                        "type": "reasoning",
                        "content": "Private model deliberation that must be discarded."
                    },
                    {"type": "message",
                            "content": "[NHÃN: AN TOÀN]\n1. KẾT LUẬN CHI TIẾT: An toàn.",
                            "reasoning_content": "Legitimate communication."
                    }
                ]
            }
            mock_query.return_value = mock_response

            email_text = "Subject: University Seminar Notice. Dear students, you are invited to a guest lecture on Cyber Security next Monday."
            host = "http://localhost:1234"

            content, reasoning = analyze_with_lm_studio(
                email_text=email_text,
                host=host,
                layers_summary="Layer 1: Safe\nLayer 2: Safe",
                threat_info={"vector": "University Seminar", "severity": "Safe"}
            )

            # Verify return values
            self.assertIn("[NHÃN: AN TOÀN]", content)
            self.assertEqual(reasoning, "")

            # Verify query_lm_studio was called with correct arguments
            mock_query.assert_called_once()
            args, kwargs = mock_query.call_args

            self.assertEqual(args[0], host)
            self.assertEqual(args[1], "api/v1/chat")

            payload = args[2]
            self.assertEqual(payload["max_output_tokens"], 1024)
            self.assertEqual(payload["reasoning"], "off")
            self.assertFalse(payload["store"])

            # Assert system prompt properties
            system_content = payload["system_prompt"]
            self.assertIn("highly objective", system_content)
            self.assertIn("false alarms", system_content)
            self.assertIn("university seminar announcements", system_content)
            self.assertIn("[NHÃN: LỪA ĐẢO]", system_content)
            self.assertIn("credential harvesting", system_content)
            self.assertIn("social engineering", system_content)

            # Assert user prompt properties
            user_content = payload["input"]
            self.assertIn("University Seminar Notice", user_content)
            self.assertIn("THAM KHẢO LỊCH SỬ", user_content)
            self.assertIn("TUYỆT ĐỐI KHÔNG ĐƯỢC trộn lẫn", user_content)
            self.assertIn("RÚT KINH NGHIỆM", user_content)

            app_source = Path("app.py").read_text(encoding="utf-8")
            self.assertNotIn("AI Thought Process", app_source)
            self.assertNotIn('"reasoning_summary"', app_source)

    # 23. Test advanced_cleaning_defense text truncation to 15,000 characters
    def test_advanced_cleaning_defense_truncation(self):
        from src.preprocessing import advanced_cleaning_defense
        long_text = "a" * 20000
        cleaned = advanced_cleaning_defense(long_text)
        self.assertEqual(len(cleaned), 15000)

    # 24. Test that the train pipeline configurations and bilingual stop words exist
    def test_train_pipeline_bilingual_elements(self):
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        vietnamese_stop_words = ['và', 'của', 'là', 'trong', 'để', 'tự', 'cho', 'này', 'các', 'bạn', 'có', 'một', 'được', 'với', 'những', 'ra', 'thì', 'khi', 'vào', 'đã', 'đang', 'sẽ', 'tại', 'nhà', 'nơi', 'như', 'nhưng', 'cũng', 'đến', 'ở', 'đó', 'nếu', 'nào', 'qua', 'bằng', 'chỉ', 'từ', 'lên', 'đi', 'về', 'lại', 'mới', 'cả', 'ngày', 'năm', 'sau', 'trước', 'kia', 'họ', 'tôi', 'anh', 'chị', 'em', 'ông', 'bà', 'nó', 'chúng', 'đây', 'đâu', 'sao', 'thế', 'muốn', 'luôn', 'rất', 'quá', 'nhất', 'nhiều', 'ít', 'toàn', 'hoặc', 'vì', 'nên', 'nhờ', 'nữa', 'nọ', 'sự', 'việc', 'cái', 'con', 'mọi', 'mọi', 'mỗi', 'từng', 'vừa', 'ngay', 'thường', 'hay', 'chưa', 'không', 'chẳng', 'ai', 'gì']
        bilingual_stop_words = list(ENGLISH_STOP_WORDS) + vietnamese_stop_words
        
        self.assertTrue(len(bilingual_stop_words) > len(ENGLISH_STOP_WORDS))
        self.assertIn("và", bilingual_stop_words)
        self.assertIn("the", bilingual_stop_words)

    # 25. Test brand domain extraction helper
    def test_extract_main_domain_part(self):
        from src.preprocessing import extract_main_domain_part
        self.assertEqual(extract_main_domain_part("shopee.vn"), "shopee")
        self.assertEqual(extract_main_domain_part("vietcombank.com.vn"), "vietcombank")
        self.assertEqual(extract_main_domain_part("hust.edu.vn"), "hust")
        self.assertEqual(extract_main_domain_part("sub.google.com"), "google")
        self.assertEqual(extract_main_domain_part("google.com"), "google")
        self.assertEqual(extract_main_domain_part(""), "")

    # 26. Test homoglyphs normalization helper
    def test_normalize_homoglyphs(self):
        from src.preprocessing import normalize_homoglyphs
        self.assertEqual(normalize_homoglyphs("g00g1e"), "google")
        self.assertEqual(normalize_homoglyphs("paypa1"), "paypal")
        self.assertEqual(normalize_homoglyphs("sh0p33"), "shopee")
        self.assertEqual(normalize_homoglyphs("faceb00k"), "facebook")
        self.assertEqual(normalize_homoglyphs("vvord"), "word")
        self.assertEqual(normalize_homoglyphs("rnail"), "mail")
        self.assertEqual(normalize_homoglyphs(""), "")

    # 27. Test Levenshtein distance helper
    def test_levenshtein_distance(self):
        from src.preprocessing import levenshtein_distance
        self.assertEqual(levenshtein_distance("cat", "dog"), 3)
        self.assertEqual(levenshtein_distance("cat", "cot"), 1)
        self.assertEqual(levenshtein_distance("google", "goolge"), 2)
        self.assertEqual(levenshtein_distance("", "abc"), 3)

    # 28. Test lookalike and typosquatting detection
    def test_check_lookalike_and_typosquatting(self):
        from src.preprocessing import check_lookalike_and_typosquatting
        # Safe / Genuine
        is_la, desc = check_lookalike_and_typosquatting("google.com")
        self.assertFalse(is_la)
        
        # Homoglyphs
        is_la, desc = check_lookalike_and_typosquatting("g00g1e.com")
        self.assertTrue(is_la)
        self.assertIn("Homoglyph Attack", desc)
        
        # Typosquatting
        is_la, desc = check_lookalike_and_typosquatting("goolge.com")
        self.assertTrue(is_la)
        self.assertIn("Typosquatting", desc)
        
        # Substring/Appended words
        is_la, desc = check_lookalike_and_typosquatting("secure-google.com")
        self.assertTrue(is_la)
        self.assertIn("chứa thương hiệu chính thức", desc)

    # 29. Dashboard must not perform direct HTTP probes, even after a UI opt-in.
    def test_check_url_activity_is_offline_even_when_requested(self):
        from src.preprocessing import check_url_activity
        res = check_url_activity("google.com", allow_network=True)
        self.assertFalse(res['active'])
        self.assertIsNone(res['status_code'])
        self.assertIn("isolated egress-scanner", res['error'])

    # 30. A loopback target cannot cause a network request either.
    def test_check_url_activity_rejects_loopback_without_network(self):
        from src.preprocessing import check_url_activity
        res = check_url_activity("http://127.0.0.1:8501/admin", allow_network=True)
        self.assertFalse(res['active'])
        self.assertIn("unavailable", res['error'])

    # 31. Test URL risk analysis cache functionality
    @patch("src.preprocessing.check_url_activity")
    def test_analyze_url_risk_caching(self, mock_check_activity):
        mock_check_activity.return_value = {
            "active": True,
            "status_code": 200,
            "final_url": "https://randomdomain123.com",
            "error": None
        }
        
        from src.preprocessing import analyze_url_risk
        analyze_url_risk.cache_clear()
        
        analyze_url_risk("https://randomdomain123.com", check_activity=True)
        analyze_url_risk("https://randomdomain123.com", check_activity=True)
        
        mock_check_activity.assert_called_once_with(
            "https://randomdomain123.com",
            allow_network=True,
        )

    @patch("src.preprocessing.check_url_activity")
    def test_analyze_url_risk_is_offline_by_default(self, mock_check_activity):
        from src.preprocessing import analyze_url_risk

        analyze_url_risk.cache_clear()
        analyze_url_risk("https://randomdomain124.com")
        mock_check_activity.assert_not_called()

    def test_url_activity_blocks_private_destinations(self):
        from src.preprocessing import check_url_activity
        result = check_url_activity("http://127.0.0.1:8501/admin", allow_network=True)
        self.assertFalse(result["active"])
        self.assertIn("unavailable", result["error"])

    def test_mime_attachment_is_inspected(self):
        raw_eml = """From: sender@example.com
Subject: Monthly update
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="b"

--b
Content-Type: text/plain

Please see the attachment.
--b
Content-Type: application/octet-stream; name="invoice.exe"
Content-Disposition: attachment; filename="invoice.exe"
Content-Transfer-Encoding: base64

TVqQAAMAAAAEAAAA
--b--
"""
        parsed = parse_raw_email(raw_eml)
        self.assertEqual(parsed["attachments"][0]["filename"], "invoice.exe")
        layers = evaluate_12_layers(raw_eml, parsed)
        attachment_layer = next(layer for layer in layers if layer["layer"].startswith("10."))
        self.assertIn("nguy hại", attachment_layer["status"])

    def test_llm_label_must_be_the_first_non_whitespace_content(self):
        from src.preprocessing import extract_llm_label
        self.assertEqual(extract_llm_label("[NHÃN: AN TOÀN]\nBình thường."), "SAFE")
        self.assertEqual(extract_llm_label("Phân tích: [NHÃN: AN TOÀN]"), "UNKNOWN")
        self.assertEqual(extract_llm_label("[NHÃN: LỪA ĐẢO]\nCó link giả."), "PHISHING")

    def test_csv_formula_escape(self):
        from src.preprocessing import escape_csv_formula
        self.assertEqual(escape_csv_formula("=HYPERLINK(\"https://evil\")"), "'=HYPERLINK(\"https://evil\")")
        self.assertEqual(escape_csv_formula("normal email"), "normal email")

    # 32. Test security refinements
    def test_security_refinements_improved(self):
        from src.preprocessing import (
            normalize_homoglyphs,
            check_lookalike_and_typosquatting,
            extract_main_domain_part
        )
        
        # A. Cyrillic homoglyph spoofing: Cyrillic 'о' (U+043E) and 'е' (U+0435)
        # 'gооеgle.com' contains Cyrillic characters and normalizes to Latin 'gooegle'
        cyr_domain = "gооеgle.com"
        norm_cyr = normalize_homoglyphs("gооеgle")
        self.assertEqual(norm_cyr, "gooegle")
        
        is_la, desc = check_lookalike_and_typosquatting(cyr_domain)
        self.assertTrue(is_la)
        self.assertIn("Typosquatting", desc)
        
        # Test visual homoglyph spoofing where the brand name matches exactly after mapping
        # 'gооglе.com' (with Cyrillic 'о' and Cyrillic 'е') normalizes to Latin 'google'
        cyr_exact_domain = "gооglе.com"
        norm_cyr_exact = normalize_homoglyphs("gооglе")
        self.assertEqual(norm_cyr_exact, "google")
        
        is_la, desc = check_lookalike_and_typosquatting(cyr_exact_domain)
        self.assertTrue(is_la)
        self.assertIn("Homoglyph Attack", desc)
        
        # B. Concatenated brand names (e.g. shopeesecure.com, vietcombanklogin.com)
        is_la, desc = check_lookalike_and_typosquatting("shopeesecure.com")
        self.assertTrue(is_la)
        self.assertIn("thương hiệu chính thức 'shopee'", desc)
        
        is_la, desc = check_lookalike_and_typosquatting("vietcombanklogin.com")
        self.assertTrue(is_la)
        self.assertIn("thương hiệu chính thức 'vietcombank'", desc)
        
        # C. Regional brand variations (e.g. microsoft.com.vn, google.com.vn, etc.) should not be flagged
        is_la, desc = check_lookalike_and_typosquatting("microsoft.com.vn")
        self.assertFalse(is_la)
        
        is_la, desc = check_lookalike_and_typosquatting("google.com.vn")
        self.assertFalse(is_la)
        
        is_la, desc = check_lookalike_and_typosquatting("shopee.com.vn")
        self.assertFalse(is_la)

        # D. Double TLD (e.g., shopee.co.uk brand extraction)
        self.assertEqual(extract_main_domain_part("shopee.co.uk"), "shopee")
        
        # E. Short brand concatenation avoidance (e.g., hustler.com vs hust.edu.vn)
        is_la, desc = check_lookalike_and_typosquatting("hustler.com")
        self.assertFalse(is_la)

    # 33. Test LLM label extraction logic to prevent false consensus alerts
    def test_llm_label_extraction_logic(self):
        def extract_label(content):
            llm_label = "SAFE"
            pos_phishing = content.find("[NHÃN: LỪA ĐẢO]")
            pos_safe = content.find("[NHÃN: AN TOÀN]")
            
            if pos_phishing != -1 and (pos_safe == -1 or pos_phishing < pos_safe):
                llm_label = "PHISHING"
            elif pos_safe != -1 and (pos_phishing == -1 or pos_safe < pos_phishing):
                llm_label = "SAFE"
            else:
                first_chars = content[:150].upper()
                if "LỪA ĐẢO" in first_chars or "PHISHING" in first_chars:
                    llm_label = "PHISHING"
            return llm_label

        # Test Case 1: Safe label first, phishing label later as a negation
        content_1 = "[NHÃN: AN TOÀN]\n\nEmail này an toàn vì không có dấu hiệu lừa đảo trực tiếp, không nên phân loại email này thành [NHÃN: LỪA ĐẢO]."
        self.assertEqual(extract_label(content_1), "SAFE")

        # Test Case 2: Phishing label first
        content_2 = "[NHÃN: LỪA ĐẢO]\n\nCảnh báo: email này chứa link độc hại [NHÃN: LỪA ĐẢO]."
        self.assertEqual(extract_label(content_2), "PHISHING")

        # Test Case 3: Fallback phishing keyword in first 150 characters
        content_3 = "Cảnh báo khẩn cấp: Email này có hành vi lừa đảo người dùng và đánh cắp mật khẩu."
        self.assertEqual(extract_label(content_3), "PHISHING")

        # Test Case 4: Fallback safe keyword
        content_4 = "Đây là thông báo bình thường từ quản trị viên."
        self.assertEqual(extract_label(content_4), "SAFE")

    def test_bilingual_threat_vectors(self):
        # Verify bilingual names are returned for main threat categories
        raw_text = "Chào bạn, vui lòng click vào link này để nhập mật khẩu login tài khoản ngân hàng."
        parsed_info = parse_raw_email(raw_text)
        parsed_info["links"] = ["http://fake-link.com"]
        
        # Test Credential Harvesting (ML prob >= 0.5)
        threat = classify_cyber_threat(raw_text, parsed_info, [], ml_probability=0.8)
        self.assertEqual(threat["vector"], "1. Thu thập thông tin xác thực (Credential Harvesting)")
        
        # Test BEC (ML prob >= 0.5, containing CEO fraud keyword)
        raw_bec = "Chuyển khoản khẩn cấp cho CEO ngay bây giờ."
        parsed_bec = parse_raw_email(raw_bec)
        threat_bec = classify_cyber_threat(raw_bec, parsed_bec, [], ml_probability=0.8)
        self.assertEqual(threat_bec["vector"], "2. Giả mạo email doanh nghiệp (Business Email Compromise - BEC)")
        
    def test_layer_12_dynamic_status_updates(self):
        raw_text = "Chào bạn, hãy click vào link này để xác nhận tài khoản khẩn cấp."
        parsed_info = parse_raw_email(raw_text)
        
        # 1. No LLM content, but triggered by urgency
        layers_waiting = evaluate_12_layers(raw_text, parsed_info, ml_probability=0.1, llm_content=None)
        l12_waiting = next(l for l in layers_waiting if "12. Deep Semantic AI Layer" in l["layer"])
        self.assertEqual(l12_waiting["status"], "🤖 Chờ LLM Phân tích")
        
        # 2. LLM classified as Phishing
        layers_phishing = evaluate_12_layers(raw_text, parsed_info, ml_probability=0.1, llm_content="[NHÃN: LỪA ĐẢO]\nCảnh báo lừa đảo.")
        l12_phishing = next(l for l in layers_phishing if "12. Deep Semantic AI Layer" in l["layer"])
        self.assertEqual(l12_phishing["status"], "🚨 Đỏ (Lừa đảo)")
        self.assertEqual(l12_phishing["detail"], "Mô hình LLM phân tích ngữ nghĩa phát hiện các yếu tố lừa đảo, thao túng tâm lý.")
        
        # 3. LLM classified as Safe
        layers_safe = evaluate_12_layers(raw_text, parsed_info, ml_probability=0.1, llm_content="[NHÃN: AN TOÀN]\nEmail bình thường.")
        l12_safe = next(l for l in layers_safe if "12. Deep Semantic AI Layer" in l["layer"])
        self.assertEqual(l12_safe["status"], "🟢 An toàn (Đã kiểm duyệt)")
        self.assertEqual(l12_safe["detail"], "Mô hình LLM xác nhận nội dung email an toàn sau khi phân tích sâu ngữ nghĩa.")

    def test_feedback_registry_deduplicates_and_quarantines_conflicts(self):
        from src.feedback_store import FeedbackStore

        with TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.sqlite3")
            created = store.submit(
                body="Please reset your account password now.", label=1,
                source="user_confirmed", status="accepted", llm_analysis="Generic credential theft rule."
            )
            duplicate = store.submit(
                body="  please   reset your ACCOUNT password now. ", label=1,
                source="user_confirmed", status="accepted"
            )
            conflict = store.submit(
                body="Please reset your account password now.", label=0,
                source="user_confirmed", status="accepted"
            )

            self.assertEqual(created["action"], "created")
            self.assertEqual(duplicate["action"], "duplicate")
            self.assertEqual(conflict["action"], "conflict")
            self.assertEqual(store.stats()["conflict"], 2)
            self.assertTrue(store.accepted_for_training()[0].empty)
            self.assertTrue(store.accepted_for_rag().empty)

    def test_rag_uses_only_accepted_governed_feedback(self):
        from src.feedback_store import FeedbackStore
        from src.rag_memory import retrieve_similar_errors

        with TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.sqlite3")
            store.submit(
                body="Urgent verify login account through a suspicious link",
                label=1, source="user_confirmed", status="accepted",
                llm_analysis="Quy tắc: Yêu cầu xác minh đăng nhập qua liên kết lạ cần được coi là rủi ro cao.",
            )
            store.submit(
                body="Urgent verify login account through a suspicious link two",
                label=1, source="legacy_import", status="pending",
                llm_analysis="LEGACY_SECRET_SHOULD_NOT_APPEAR",
            )
            with patch("src.rag_memory.FeedbackStore", return_value=store):
                context = retrieve_similar_errors("Urgent verify login account through a suspicious link", k=2)
            self.assertIn("BÀI HỌC ĐÃ XÁC MINH", context)
            self.assertIn("xác minh đăng nhập", context)
            self.assertNotIn("LEGACY_SECRET_SHOULD_NOT_APPEAR", context)
            self.assertNotIn("Urgent verify login account through a suspicious link", context)

    def test_llm_only_feedback_reaches_rag_but_not_ml_training(self):
        from src.feedback_store import FeedbackStore, LLM_FEEDBACK_SOURCE

        with TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.sqlite3")
            store.submit(
                body="Monthly newsletter from a verified sender with no action requested",
                label=0,
                source=LLM_FEEDBACK_SOURCE,
                status="accepted",
                llm_analysis="Ca tương tự đã được người dùng xác nhận là an toàn.",
            )

            training_rows, holdout_rows = store.accepted_for_training()
            rag_rows = store.accepted_for_rag()
            self.assertTrue(training_rows.empty)
            self.assertTrue(holdout_rows.empty)
            self.assertEqual(len(rag_rows), 1)
            self.assertEqual(rag_rows.iloc[0]["source"], LLM_FEEDBACK_SOURCE)
            self.assertEqual(rag_rows.iloc[0]["label"], 0)

    def test_duplicate_feedback_is_upgraded_with_new_llm_lesson(self):
        from src.feedback_store import (
            FeedbackStore,
            ML_AND_LLM_FEEDBACK_SOURCE,
            ML_FEEDBACK_SOURCE,
        )

        with TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.sqlite3")
            store.submit(
                body="Verified delivery-completion notification from a legitimate sender",
                label=0,
                source=ML_FEEDBACK_SOURCE,
                status="accepted",
            )
            updated = store.submit(
                body="Verified delivery-completion notification from a legitimate sender",
                label=0,
                source=ML_AND_LLM_FEEDBACK_SOURCE,
                status="accepted",
                llm_analysis="Ca tương tự đã được người dùng xác nhận là an toàn.",
            )

            training_rows, holdout_rows = store.accepted_for_training()
            rag_rows = store.accepted_for_rag()
            self.assertEqual(updated["action"], "updated")
            self.assertEqual(len(training_rows) + len(holdout_rows), 1)
            self.assertEqual(len(rag_rows), 1)
            self.assertEqual(rag_rows.iloc[0]["source"], ML_AND_LLM_FEEDBACK_SOURCE)
            self.assertIn("người dùng xác nhận", rag_rows.iloc[0]["llm_analysis"])

    def test_pending_feedback_cannot_mutate_an_accepted_lesson(self):
        from src.feedback_store import (
            FeedbackStore,
            ML_AND_LLM_FEEDBACK_SOURCE,
            ML_FEEDBACK_SOURCE,
        )

        with TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.sqlite3")
            store.submit(
                body="Verified supplier invoice with matching purchase order",
                label=0,
                source=ML_FEEDBACK_SOURCE,
                status="accepted",
                llm_analysis="Đã kiểm tra chứng từ hợp lệ.",
            )
            outcome = store.submit(
                body="Verified supplier invoice with matching purchase order",
                label=0,
                source=ML_AND_LLM_FEEDBACK_SOURCE,
                status="pending",
                llm_analysis="Nội dung chưa được analyst xác minh.",
            )

            self.assertEqual(outcome["action"], "duplicate")
            rag_rows = store.accepted_for_rag()
            self.assertEqual(rag_rows.iloc[0]["source"], ML_FEEDBACK_SOURCE)
            self.assertIn("chứng từ hợp lệ", rag_rows.iloc[0]["llm_analysis"])
            self.assertNotIn("chưa được analyst", rag_rows.iloc[0]["llm_analysis"])

    def test_feedback_governance_summary_separates_ml_and_rag_scope(self):
        from src.feedback_store import FeedbackStore, LLM_FEEDBACK_SOURCE, ML_FEEDBACK_SOURCE

        with TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.sqlite3")
            store.submit(body="safe vendor update", label=0, source=ML_FEEDBACK_SOURCE, status="accepted")
            store.submit(body="safe event newsletter", label=0, source=LLM_FEEDBACK_SOURCE, status="accepted")
            store.submit(body="pending case", label=1, source=ML_FEEDBACK_SOURCE, status="pending")

            summary = store.governance_summary()
            self.assertEqual(summary["pending"], 1)
            self.assertEqual(summary["accepted_trainable"], 1)
            self.assertEqual(summary["accepted_rag"], 2)
            self.assertEqual(summary["accepted_llm_only"], 1)

    def test_quality_gate_rejects_more_false_negatives(self):
        from src.train import _quality_gate

        baseline = {"samples": 200, "false_negatives": 2, "recall": 0.98, "precision": 0.97, "f1_score": 0.975}
        candidate = {"samples": 200, "false_negatives": 3, "recall": 0.97, "precision": 0.98, "f1_score": 0.975}
        gate = _quality_gate(candidate, baseline)
        self.assertFalse(gate["passed"])
        self.assertTrue(any(check["name"] == "không tăng bỏ lọt phishing" and not check["passed"] for check in gate["checks"]))

    def test_registry_never_promotes_a_rejected_candidate(self):
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import src.model_registry as registry

        with TemporaryDirectory() as directory:
            root = Path(directory) / "models"
            root.mkdir()
            vectorizer = TfidfVectorizer().fit(["safe meeting", "verify password"])
            model = LogisticRegression().fit(vectorizer.transform(["safe meeting", "verify password"]), [0, 1])
            active_model = root / "phishing_model.pkl"
            active_vector = root / "tfidf_vectorizer.pkl"
            active_metadata = root / "model_metadata.json"
            joblib.dump(model, active_model)
            joblib.dump(vectorizer, active_vector)
            active_metadata.write_text(json.dumps({"f1_score": 0.9}), encoding="utf-8")
            original_bytes = active_model.read_bytes()

            with patch.multiple(
                registry,
                MODELS_DIR=root,
                REGISTRY_PATH=root / "model_registry.json",
                VERSIONS_DIR=root / "versions",
                CANDIDATES_DIR=root / "candidates",
                ACTIVE_MODEL=active_model,
                ACTIVE_VECTOR=active_vector,
                ACTIVE_METADATA=active_metadata,
            ):
                legacy = registry.bootstrap_active_model()
                candidate = registry.save_candidate(vectorizer, model, {"quality_gate": {"passed": False}})
                with self.assertRaises(ValueError):
                    registry.promote_candidate(candidate["candidate_id"])
                approved = registry.save_candidate(vectorizer, model, {"quality_gate": {"passed": True}})
                promoted = registry.promote_candidate(approved["candidate_id"])
                self.assertTrue(promoted["version_id"].startswith("model-"))
                restored = registry.rollback_active_model(legacy["version_id"])
                self.assertEqual(restored["version_id"], legacy["version_id"])
            self.assertEqual(active_model.read_bytes(), original_bytes)

    def test_candidate_training_uses_immutable_benchmark_and_does_not_promote_without_baseline(self):
        import pandas as pd
        import src.train as training
        from src.feedback_store import FeedbackStore

        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            rows = []
            for index in range(300):
                rows.append({"body": f"routine internal meeting agenda {index}", "label": 0})
                rows.append({"body": f"urgent verify password through malicious link {index}", "label": 1})
            pd.DataFrame(rows).to_csv(data_dir / "tiny.csv", index=False)
            store = FeedbackStore(root / "feedback.sqlite3")
            candidate_path = root / "candidate"

            def fake_save_candidate(_vectorizer, _model, _metadata):
                return {"candidate_id": "candidate-test", "path": str(candidate_path), "status": "rejected"}

            with patch.multiple(training, DATA_DIR=data_dir, BENCHMARK_DIR=data_dir / "benchmarks"), \
                 patch("src.train.FeedbackStore", return_value=store), \
                 patch("src.train.active_model_version_id", return_value="none"), \
                 patch("src.train.load_active_artifacts", return_value=(None, None)), \
                 patch("src.train.save_candidate", side_effect=fake_save_candidate):
                result = training.run_training_pipeline("tiny.csv", auto_promote=False)

            self.assertTrue(result["success"])
            self.assertFalse(result["promoted"])
            self.assertIn("checks", result["quality_gate"])
            self.assertEqual(result["metadata"]["training_policy"]["mode"], "full_rebuild_from_selected_dataset")
            self.assertEqual(result["metadata"]["dataset_manifest"]["files"][0]["filename"], "tiny.csv")
            self.assertTrue(result["metadata"]["benchmark_lifecycle"]["immutable"])
            self.assertTrue((data_dir / "benchmarks").exists())
            self.assertTrue((candidate_path / "benchmark_missed_cases.csv").exists())

    def test_bilingual_language_routing_and_ml_security_tokens(self):
        from src.preprocessing import build_ml_document, detect_email_language, detect_email_language_details

        self.assertEqual(detect_email_language("Vui lòng xác minh tài khoản và mật khẩu của bạn"), "vi")
        self.assertEqual(detect_email_language("Please verify your account password before the meeting"), "en")
        self.assertEqual(detect_email_language("Vui lòng xác minh tài khoản và verify your account mật khẩu ngay"), "mixed")
        document = build_ml_document(
            "Khẩn cấp: hãy đăng nhập tài khoản tại http://192.0.2.1/login"
        )
        self.assertIn("lang_vi", document)
        self.assertIn("url_present", document)
        self.assertIn("url_ip_literal", document)
        self.assertIn("credential_request", document)
        self.assertIn("urgency_signal", document)
        unaccented = detect_email_language_details("Xin chao, vui long xac minh tai khoan ngan hang cua ban ngay")
        self.assertEqual(unaccented["language"], "vi")
        self.assertGreaterEqual(unaccented["confidence"], 0.75)
        # Common unaccented English tokens must not be enough to route an
        # ordinary English message through the Vietnamese response path.
        self.assertEqual(
            detect_email_language("This is a long project update for the team"),
            "en",
        )
        # Spanish accents are not evidence of Vietnamese.  This protects the
        # language route for the independent Spanish SpaPhish evaluation set.
        self.assertEqual(
            detect_email_language("Atenci\u00f3n: su cuenta ha sido bloqueada. Verifique ahora."),
            "unknown",
        )
        short_unknown = detect_email_language_details("ref: 2026-08-10")
        self.assertEqual(short_unknown["language"], "unknown")
        self.assertLess(short_unknown["confidence"], 0.5)

    def test_eml_forensics_extracts_addresses_and_dkim_evidence(self):
        from src.preprocessing import group_network_security_features

        raw_eml = """From: Finance <billing@example.com>
To: analyst@example.org
Reply-To: payments@reply.example
Return-Path: <bounce@example.com>
Authentication-Results: gateway.example; spf=pass; dkim=pass; dmarc=pass
DKIM-Signature: v=1; a=rsa-sha256; d=example.com; s=selector2026;
 bh=abc; b=def
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8

Monthly invoice attached.
"""
        parsed = parse_raw_email(raw_eml)
        self.assertEqual(parsed["addresses"]["to"], ["analyst@example.org"])
        self.assertEqual(parsed["addresses"]["reply_to"], ["payments@reply.example"])
        signature = parsed["digital_signature"]
        self.assertTrue(signature["present"])
        self.assertEqual(signature["domain"], "example.com")
        self.assertEqual(signature["selector"], "selector2026")
        self.assertEqual(signature["verification"], "not_independently_verified")
        network = group_network_security_features(raw_eml, allow_dns=False)
        self.assertEqual(network["authentication"]["header_status"]["dmarc"], "pass")
        self.assertEqual(network["dns_records"], {"A": [], "MX": [], "TXT": []})

    def test_ioc_profile_collects_domains_ips_and_signature_formats_without_trusting_them(self):
        from src.preprocessing import extract_iocs

        raw = """From: Service <notice@xn--b-nia.example>
To: analyst@example.org
Reply-To: reply@payments.example
Received: from relay.evil.example (8.8.8.8) by mx.example.org
ARC-Seal: i=1; a=rsa-sha256; d=example.org; s=arc2026; b=abc
Content-Type: multipart/signed; protocol="application/pkcs7-signature"

Review https://login.evil.example/reset and https://[2001:db8::1]/login.
-----BEGIN PGP SIGNED MESSAGE-----
"""
        profile = extract_iocs(raw, allow_dns=False)
        self.assertIn("notice@xn--b-nia.example", profile["addresses"]["from"])
        self.assertIn("payments.example", profile["domains"])
        self.assertIn("8.8.8.8", profile["ip_addresses"])
        self.assertIn("2001:db8::1", profile["ip_addresses"])
        self.assertFalse(profile["dns"]["lookup_performed"])
        self.assertEqual(profile["dns"]["state"], "not_queried_offline_privacy_mode")
        self.assertTrue(profile["digital_signatures"]["arc"]["present"])
        self.assertTrue(profile["digital_signatures"]["smime"]["present"])
        self.assertTrue(profile["digital_signatures"]["openpgp"]["present"])
        self.assertEqual(profile["digital_signatures"]["arc"]["verification"], "not_independently_verified")

if __name__ == "__main__":
    unittest.main()
