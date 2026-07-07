/**
 * 用户反馈收集 (Feedback Collection)
 *
 * 每个助手回复下方显示赞/踩按钮，
 * 收集用户偏好用于后续 DPO 迭代。
 */

class FeedbackCollector {
    constructor() {
        this.apiBase = window.location.origin;
        this.initialized = false;
    }

    /**
     * 为消息元素添加反馈按钮
     * @param {HTMLElement} messageElement - 助手消息的 DOM 元素
     * @param {number} messageId - 消息在数据库中的 ID
     */
    attachFeedbackButtons(messageElement, messageId) {
        if (!messageElement || messageId === undefined) return;

        // 检查是否已有反馈按钮
        if (messageElement.querySelector('.feedback-buttons')) return;

        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'feedback-buttons';
        feedbackDiv.innerHTML = `
            <button class="feedback-btn like-btn" data-rating="like" title="回答不错">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/>
                    <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
                </svg>
            </button>
            <button class="feedback-btn dislike-btn" data-rating="dislike" title="回答不好">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/>
                    <path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/>
                </svg>
            </button>
        `;

        // 点击事件
        feedbackDiv.querySelector('.like-btn').addEventListener('click', (e) => {
            this.submitFeedback(messageId, 'like', e.target.closest('.feedback-btn'));
        });

        feedbackDiv.querySelector('.dislike-btn').addEventListener('click', (e) => {
            this.submitFeedback(messageId, 'dislike', e.target.closest('.feedback-btn'));
        });

        messageElement.appendChild(feedbackDiv);
    }

    /**
     * 提交反馈
     */
    async submitFeedback(messageId, rating, buttonElement) {
        try {
            const response = await fetch(`${this.apiBase}/v1/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message_id: messageId,
                    rating: rating,
                    comment: '',
                }),
            });

            if (response.ok) {
                this.showFeedbackConfirmation(buttonElement, rating);
            }
        } catch (error) {
            console.error('Feedback submission failed:', error);
        }
    }

    /**
     * 显示反馈确认动画
     */
    showFeedbackConfirmation(button, rating) {
        const container = button.parentElement;
        if (!container) return;

        // 禁用所有按钮
        container.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = '0.5';
            btn.style.cursor = 'default';
        });

        // 高亮选中按钮
        button.style.opacity = '1';
        button.style.color = rating === 'like' ? '#7EC87B' : '#E74C3C';
        button.style.transform = 'scale(1.2)';
        setTimeout(() => {
            button.style.transform = 'scale(1)';
        }, 200);
    }
}

// 全局实例
window.feedbackCollector = new FeedbackCollector();
